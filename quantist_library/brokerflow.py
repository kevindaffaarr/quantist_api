from __future__ import annotations
from typing import Literal
import gc

import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sqlalchemy.sql import func

import database as db
import dependencies as dp
from quantist_library import genchart

class StockBFFull():
	"""
	Init, Calculate Indicators, and Get Chart for Broker Flow Methods
	"""
	def __init__(self,
		stockcode: str | None = None,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		clustering_method: dp.ClusteringMethod = dp.ClusteringMethod.correlation,
		n_selected_cluster:int | None = None,
		period_mf: int | None = None,
		period_prop: int | None = None,
		period_pricecorrel: int | None = None,
		period_mapricecorrel: int | None = None,
		period_vwap:int | None = None,
		pow_high_prop: int | None = None,
		pow_high_pricecorrel: int | None = None,
		pow_high_mapricecorrel: int | None = None,
		pow_medium_prop: int | None = None,
		pow_medium_pricecorrel: int | None = None,
		pow_medium_mapricecorrel: int | None = None,
		training_start_index: float | None = None,
		training_end_index: float | None = None,
		min_n_cluster: int | None = None,
		max_n_cluster: int | None = None,
		splitted_min_n_cluster: int | None = None,
		splitted_max_n_cluster: int | None = None,
		stepup_n_cluster_threshold: float | None = None,
		dbs: db.Session = next(db.get_dbs()),
		) -> None:

		self.stockcode = stockcode
		self.startdate = startdate
		self.enddate = enddate
		self.clustering_method = clustering_method
		self.n_selected_cluster = n_selected_cluster
		self.period_mf = period_mf
		self.period_prop = period_prop
		self.period_pricecorrel = period_pricecorrel
		self.period_mapricecorrel = period_mapricecorrel
		self.period_vwap = period_vwap
		self.pow_high_prop = pow_high_prop
		self.pow_high_pricecorrel = pow_high_pricecorrel
		self.pow_high_mapricecorrel = pow_high_mapricecorrel
		self.pow_medium_prop = pow_medium_prop
		self.pow_medium_pricecorrel = pow_medium_pricecorrel
		self.pow_medium_mapricecorrel = pow_medium_mapricecorrel
		self.training_start_index = training_start_index
		self.training_end_index = training_end_index
		self.min_n_cluster = min_n_cluster
		self.max_n_cluster = max_n_cluster
		self.splitted_min_n_cluster = splitted_min_n_cluster
		self.splitted_max_n_cluster = splitted_max_n_cluster
		self.stepup_n_cluster_threshold = stepup_n_cluster_threshold
		self.dbs = dbs

		self.preoffset_startdate = None

		self.selected_broker = None
		self.optimum_n_selected_cluster = None
		self.optimum_corr = None
		self.wf_indicators:pd.DataFrame = pd.DataFrame()

	async def fit(self) -> StockBFFull:
		# Get default bf params
		default_bf = await self.__get_default_bf(dbs=self.dbs)
		assert isinstance(self.stockcode, str), "stockcode must be string"
		assert isinstance(self.training_start_index, float), "training_start_index must be float"
		assert isinstance(self.training_end_index, float), "training_end_index must be float"
		assert isinstance(self.splitted_min_n_cluster, int), "splitted_min_n_cluster must be int"
		assert isinstance(self.splitted_max_n_cluster, int), "splitted_max_n_cluster must be int"
		assert isinstance(self.stepup_n_cluster_threshold, float), "stepup_n_cluster_threshold must be int"
		assert isinstance(self.period_mf, int), "period_mf must be int"
		assert isinstance(self.period_prop, int), "period_prop must be int"
		assert isinstance(self.period_pricecorrel, int), "period_pricecorrel must be int"
		assert isinstance(self.period_mapricecorrel, int), "period_mapricecorrel must be int"
		assert isinstance(self.period_vwap, int), "period_vwap must be int"
		assert isinstance(self.pow_high_prop, int), "pow_high_prop must be int"
		assert isinstance(self.pow_high_pricecorrel, int), "pow_high_pricecorrel must be int"
		assert isinstance(self.pow_high_mapricecorrel, int), "pow_high_mapricecorrel must be int"
		assert isinstance(self.pow_medium_prop, int), "pow_medium_prop must be int"
		assert isinstance(self.pow_medium_pricecorrel, int), "pow_medium_pricecorrel must be int"
		assert isinstance(self.pow_medium_mapricecorrel, int), "pow_medium_mapricecorrel must be int"

		# Check Does Stock Code is composite
		if self.stockcode == 'composite':
			raise ValueError("Broker Flow is not available yet for index")
		# Check Does Stock Code is available in database
		else:
			qry = self.dbs.query(db.ListStock.code).filter(db.ListStock.code == self.stockcode)
			row = pd.read_sql(sql=qry.statement, con=self.dbs.bind)
			if len(row) == 0:
				raise KeyError("There is no such stock code in the database.")

		# Get full stockdatatransaction
		raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumvol, raw_data_broker_sumval = \
			await self.__get_stock_raw_data(
				stockcode=self.stockcode,
				startdate=self.startdate,
				enddate=self.enddate,
				preoffset_period_param=self.preoffset_period_param,
				dbs=self.dbs
				)
		
		if self.clustering_method == dp.ClusteringMethod.timeseries:
			# Get broker flow parameters using timeseries method
			self.selected_broker, self.optimum_n_selected_cluster, self.optimum_corr, self.broker_cluster, self.broker_ncum = \
				await self.__get_timeseries_bf_parameter(
					raw_data_close=raw_data_full["close"],
					raw_data_broker_nval=raw_data_broker_nval,
					splitted_min_n_cluster=self.splitted_min_n_cluster,
					splitted_max_n_cluster=self.splitted_max_n_cluster,
					training_start_index=self.training_start_index,
					training_end_index=self.training_end_index,
					stepup_n_cluster_threshold=self.stepup_n_cluster_threshold,
				)
			
			# Adjust plusmin of raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval
			raw_data_broker_nvol = await self.__adjust_plusmin_df(df=raw_data_broker_nvol, broker_cluster=self.broker_cluster)
			raw_data_broker_nval = await self.__adjust_plusmin_df(df=raw_data_broker_nval, broker_cluster=self.broker_cluster)
			raw_data_broker_sumval = await self.__adjust_plusmin_df(df=raw_data_broker_sumval, broker_cluster=self.broker_cluster)
		else:
			# Get broker flow parameters using correlation method
			self.selected_broker, self.optimum_n_selected_cluster, self.optimum_corr, self.broker_features = \
				await self.__get_bf_parameters(
					raw_data_close=raw_data_full["close"],
					raw_data_broker_nval=raw_data_broker_nval,
					raw_data_broker_sumval=raw_data_broker_sumval,
					n_selected_cluster=self.n_selected_cluster,
					training_start_index=self.training_start_index,
					training_end_index=self.training_end_index,
					splitted_min_n_cluster=self.splitted_min_n_cluster,
					splitted_max_n_cluster=self.splitted_max_n_cluster,
					stepup_n_cluster_threshold=self.stepup_n_cluster_threshold,
				)

		# Calc broker flow indicators
		self.wf_indicators = await self.calc_wf_indicators(
			raw_data_full = raw_data_full,
			raw_data_broker_nvol = raw_data_broker_nvol,
			raw_data_broker_nval = raw_data_broker_nval,
			raw_data_broker_sumval = raw_data_broker_sumval,
			selected_broker = self.selected_broker,
			period_mf = self.period_mf,
			period_prop = self.period_prop,
			period_pricecorrel = self.period_pricecorrel,
			period_mapricecorrel = self.period_mapricecorrel,
			period_vwap = self.period_vwap,
			pow_high_prop = self.pow_high_prop,
			pow_high_pricecorrel = self.pow_high_pricecorrel,
			pow_high_mapricecorrel = self.pow_high_mapricecorrel,
			pow_medium_prop = self.pow_medium_prop,
			pow_medium_pricecorrel = self.pow_medium_pricecorrel,
			pow_medium_mapricecorrel = self.pow_medium_mapricecorrel,
			preoffset_period_param = self.preoffset_period_param,
		)

		return self

	async def __get_default_bf(self,dbs: db.Session = next(db.get_dbs())) -> pd.Series:
		# Get Default Broker Flow
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter((db.DataParam.param.like("default_bf_%")) | \
				(db.DataParam.param.like("default_stockcode")) | \
				(db.DataParam.param.like("default_months_range")))
		default_bf = pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])
		
		# Default stock is parameterized, may become a branding or endorsement option
		self.stockcode = (str(default_bf['default_stockcode']) if self.stockcode is None else self.stockcode).lower()
		
		# Data Parameter
		self.period_mf = int(default_bf['default_bf_period_mf']) if self.period_mf is None else self.period_mf
		self.period_prop = int(default_bf['default_bf_period_prop']) if self.period_prop is None else self.period_prop
		self.period_pricecorrel = int(default_bf['default_bf_period_pricecorrel']) if self.period_pricecorrel is None else self.period_pricecorrel
		self.period_mapricecorrel = int(default_bf['default_bf_period_mapricecorrel']) if self.period_mapricecorrel is None else self.period_mapricecorrel
		self.period_vwap = int(default_bf['default_bf_period_vwap']) if self.period_vwap is None else self.period_vwap
		self.pow_high_prop = int(default_bf['default_bf_pow_high_prop']) if self.pow_high_prop is None else self.pow_high_prop
		self.pow_high_pricecorrel = int(default_bf['default_bf_pow_high_pricecorrel']) if self.pow_high_pricecorrel is None else self.pow_high_pricecorrel
		self.pow_high_mapricecorrel = int(default_bf['default_bf_pow_high_mapricecorrel']) if self.pow_high_mapricecorrel is None else self.pow_high_mapricecorrel
		self.pow_medium_prop = int(default_bf['default_bf_pow_medium_prop']) if self.pow_medium_prop is None else self.pow_medium_prop
		self.pow_medium_pricecorrel = int(default_bf['default_bf_pow_medium_pricecorrel']) if self.pow_medium_pricecorrel is None else self.pow_medium_pricecorrel
		self.pow_medium_mapricecorrel = int(default_bf['default_bf_pow_medium_mapricecorrel']) if self.pow_medium_mapricecorrel is None else self.pow_medium_mapricecorrel
		self.preoffset_period_param = max(self.period_mf,self.period_prop,self.period_pricecorrel,(self.period_mapricecorrel+self.period_pricecorrel),self.period_vwap)-1

		self.training_start_index = int(default_bf['default_bf_training_start_index'])/100 if self.training_start_index is None else self.training_start_index/100
		self.training_end_index = int(default_bf['default_bf_training_end_index'])/100 if self.training_end_index is None else self.training_end_index/100
		self.min_n_cluster = int(default_bf['default_bf_min_n_cluster']) if self.min_n_cluster is None else self.min_n_cluster
		self.max_n_cluster = int(default_bf['default_bf_max_n_cluster']) if self.max_n_cluster is None else self.max_n_cluster
		self.splitted_min_n_cluster = int(default_bf['default_bf_splitted_min_n_cluster']) if self.splitted_min_n_cluster is None else self.splitted_min_n_cluster
		self.splitted_max_n_cluster = int(default_bf['default_bf_splitted_max_n_cluster']) if self.splitted_max_n_cluster is None else self.splitted_max_n_cluster
		self.stepup_n_cluster_threshold = int(default_bf['default_bf_stepup_n_cluster_threshold'])/100 if self.stepup_n_cluster_threshold is None else self.stepup_n_cluster_threshold/100

		self.startdate = self.enddate - relativedelta(months=int(default_bf['default_months_range'])) if self.startdate is None else self.startdate
		
		return default_bf
	
	# Get Net Val Sum Val Broker Transaction
	async def __get_nvsv_broker_transaction(self,raw_data_broker_full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		# Aggretate by broker then broker to column for each net and sum
		raw_data_broker_nvol = raw_data_broker_full.pivot(index=None,columns="broker",values="nvol")
		raw_data_broker_nval = raw_data_broker_full.pivot(index=None,columns="broker",values="nval")
		raw_data_broker_sumvol = raw_data_broker_full.pivot(index=None,columns="broker",values="sumvol")
		raw_data_broker_sumval = raw_data_broker_full.pivot(index=None,columns="broker",values="sumval")

		# Fill na
		raw_data_broker_nvol.fillna(value=0, inplace=True)
		raw_data_broker_nval.fillna(value=0, inplace=True)
		raw_data_broker_sumval.fillna(value=0, inplace=True)

		# Return
		return raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumvol, raw_data_broker_sumval

	async def __get_full_broker_transaction(self,
		stockcode: str,
		preoffset_startdate: datetime.date = ...,
		enddate: datetime.date = datetime.date.today(),
		dbs: db.Session = next(db.get_dbs()),
		):
		# Query Definition
		qry = dbs.query(
			db.StockTransaction.date,
			db.StockTransaction.broker,
			db.StockTransaction.bvol,
			db.StockTransaction.svol,
			db.StockTransaction.bval,
			db.StockTransaction.sval,
			(db.StockTransaction.bvol - db.StockTransaction.svol).label("nvol"), # type: ignore
			(db.StockTransaction.bval - db.StockTransaction.sval).label("nval"), # type: ignore
			(db.StockTransaction.bvol + db.StockTransaction.svol).label("sumvol"), # type: ignore
			(db.StockTransaction.bval + db.StockTransaction.sval).label("sumval") # type: ignore
		).filter((db.StockTransaction.code == stockcode))

		# Main Query
		qry_main = qry.filter(db.StockTransaction.date.between(preoffset_startdate, enddate))\
			.order_by(db.StockTransaction.date.asc(), db.StockTransaction.broker.asc())

		# Main Query Fetching
		raw_data_broker_full = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.reset_index(drop=True).set_index("date")

		# Data Cleansing: fillna
		raw_data_broker_full.fillna(value=0, inplace=True)

		return raw_data_broker_full

	async def __get_stock_price_data(self,
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		preoffset_period_param: int = 50,
		default_months_range: int = 12,
		dbs: db.Session = next(db.get_dbs()),
		) -> tuple[pd.DataFrame, datetime.date]:

		# Define startdate if None equal to last year of enddate
		if startdate is None:
			startdate = enddate - relativedelta(months=default_months_range)

		# Query Definition
		qry = dbs.query(
			db.StockData.date,
			db.StockData.previous,
			db.StockData.openprice,
			db.StockData.high,
			db.StockData.low,
			db.StockData.close,
			db.StockData.value,
		).filter(db.StockData.code == stockcode)

		# Main Query
		qry_main = qry.filter(db.StockData.date.between(startdate, enddate)).order_by(db.StockData.date.asc())

		# Main Query Fetching
		raw_data_main = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.reset_index(drop=True).set_index('date')

		# Check how many row is returned
		if raw_data_main.shape[0] == 0:
			raise ValueError("No data available inside date range")
		
		# Update self.startdate and self.enddate to available date in database
		self.startdate = raw_data_main.index[0].date() # type: ignore
		self.enddate = raw_data_main.index[-1].date() # type: ignore

		# Pre-Data Query
		startdate = self.startdate
		qry_pre = qry.filter(db.StockData.date < startdate)\
			.order_by(db.StockData.date.desc())\
			.limit(preoffset_period_param)\
			.subquery()
		qry_pre = dbs.query(qry_pre).order_by(qry_pre.c.date.asc())

		# Pre-Data Query Fetching
		raw_data_pre = pd.read_sql(sql=qry_pre.statement, con=dbs.bind, parse_dates=["date"])\
			.reset_index(drop=True).set_index('date')

		# Concatenate Pre and Main Query
		raw_data_full = pd.concat([raw_data_pre,raw_data_main])
		
		if len(raw_data_pre) > 0:
			preoffset_startdate = raw_data_pre.index[0].date() # type: ignore
		else:
			preoffset_startdate = startdate

		# Data Cleansing: zero openprice replace with previous
		raw_data_full['openprice'] = raw_data_full['openprice'].mask(raw_data_full['openprice'].eq(0),raw_data_full['previous'])

		# End of Method: Return or Assign Attribute
		return raw_data_full, preoffset_startdate

	async def __get_stock_raw_data(self,
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_months_range: int = 12,
		preoffset_period_param: int = 50,
		dbs: db.Session = next(db.get_dbs()),
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		# Get Stockdata Full
		raw_data_full, self.preoffset_startdate = await self.__get_stock_price_data(
			stockcode=stockcode,startdate=startdate,enddate=enddate,
			preoffset_period_param=preoffset_period_param,
			default_months_range=default_months_range,dbs=dbs)

		# Get Raw Data Broker Full
		raw_data_broker_full = await self.__get_full_broker_transaction(
			stockcode=stockcode,
			preoffset_startdate=self.preoffset_startdate,
			enddate=enddate,
			dbs=dbs)

		# Transform Raw Data Broker
		raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumvol, raw_data_broker_sumval = await self.__get_nvsv_broker_transaction(raw_data_broker_full=raw_data_broker_full)

		return raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumvol, raw_data_broker_sumval

	#TODO get composite raw data def: __get_composite_raw_data ()

	async def __get_selected_broker(self,
		clustered_features: pd.DataFrame,
		centroids_cluster: pd.DataFrame,
		n_selected_cluster: int = 1,
		) -> list[str]:

		# Get index of max value in column 0 in centroid
		selected_cluster = (centroids_cluster[0]).nlargest(n_selected_cluster).index.tolist()
		# selected_cluster = (abs(centroids_cluster[0]*centroids_cluster[1])).nlargest(n_selected_cluster).index.tolist()
		
		# Get sorted selected broker
		selected_broker = clustered_features.loc[clustered_features["cluster"].isin(selected_cluster), :]\
			.sort_values(by="corr_ncum_close", ascending=False)\
			.index.tolist()

		return selected_broker

	async def __get_corr_selected_broker_ncum(self,
		clustered_features: pd.DataFrame,
		raw_data_close: pd.Series,
		broker_ncum: pd.DataFrame,
		centroids_cluster: pd.DataFrame,
		n_selected_cluster: int = 1,
		) -> float:
		selected_broker = await self.__get_selected_broker(
			clustered_features=clustered_features,
			centroids_cluster=centroids_cluster,
			n_selected_cluster=n_selected_cluster
			)

		# Get selected broker transaction by columns of net_stockdatatransaction, then sum each column to aggregate to date
		selected_broker_ncum = broker_ncum[selected_broker].sum(axis=1).rename("selected_broker_ncum")

		# Return correlation between close and selected_broker_ncum
		return selected_broker_ncum.corr(raw_data_close)

	async def __optimize_selected_cluster(self,
		clustered_features: pd.DataFrame,
		raw_data_close: pd.Series,
		broker_ncum: pd.DataFrame,
		centroids_cluster: pd.DataFrame,
		stepup_n_cluster_threshold: float = 0.05,
		n_selected_cluster: int | None = None,
		) -> tuple[list[str], int, float]:
		# Check does n_selected_cluster already defined
		if n_selected_cluster is None:
			# Define correlation param
			corr_list = []

			# Iterate optimum n_cluster
			for n_selected_cluster in range(1,len(centroids_cluster)):
				# Get correlation between close and selected_broker_ncum
				selected_broker_ncum_corr = await self.__get_corr_selected_broker_ncum(
					clustered_features,
					raw_data_close,
					broker_ncum,
					centroids_cluster,
					n_selected_cluster
					)
				# Get correlation
				corr_list.append(selected_broker_ncum_corr)

			# Define optimum n_selected_cluster
			max_corr: float = np.max(corr_list)
			index_max_corr: int = int(np.argmax(corr_list))
			optimum_corr: float = max_corr
			optimum_n_selected_cluster: int = index_max_corr + 1

			for i in range (index_max_corr):
				if (max_corr-corr_list[i]) < stepup_n_cluster_threshold:
					optimum_n_selected_cluster = i+1
					optimum_corr = corr_list[i]
					break
		# -- End of if

		# If n_selected_cluster is defined
		else:
			optimum_n_selected_cluster: int = n_selected_cluster
			optimum_corr = await self.__get_corr_selected_broker_ncum(
				clustered_features, 
				raw_data_close, 
				broker_ncum, 
				centroids_cluster, 
				n_selected_cluster
				)

		# Get Selected Broker from optimum n_selected_cluster
		selected_broker = await self.__get_selected_broker(
			clustered_features=clustered_features,
			centroids_cluster=centroids_cluster,
			n_selected_cluster=optimum_n_selected_cluster
			)

		return selected_broker, optimum_n_selected_cluster, optimum_corr

	async def __kmeans_clustering(self,
		features: pd.DataFrame,
		x: str,
		y: str,
		min_n_cluster:int = 4,
		max_n_cluster:int = 10,
		) -> tuple[pd.DataFrame, pd.DataFrame]:

		# Get X and Y
		X = features[[x,y]].values
		# Define silhouette param
		silhouette_coefficient = []
		max_n_cluster = min(max_n_cluster, len(X)-1)

		# Iterate optimum n_cluster
		for n_cluster in range(min_n_cluster, max_n_cluster+1):
			# Clustering
			kmeans = KMeans(init="k-means++", n_init='auto', n_clusters=n_cluster, random_state=0).fit(X)
			score = silhouette_score(X, kmeans.labels_)
			silhouette_coefficient.append(score)
		# Define optimum n_cluster
		optimum_n_cluster = int(np.argmax(silhouette_coefficient)) + min_n_cluster

		# Clustering with optimum n cluster
		kmeans = KMeans(init="k-means++", n_init='auto', n_clusters=optimum_n_cluster, random_state=0).fit(X)
		# Get cluster label
		features["cluster"] = kmeans.labels_
		# Get location of cluster center
		centroids_cluster = pd.DataFrame(kmeans.cluster_centers_)

		return features, centroids_cluster

	async def __xy_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
		df_std = StandardScaler().fit_transform(df)
		return pd.DataFrame(df_std, index=df.index, columns=df.columns)

	async def __get_bf_parameters(self,
		raw_data_close: pd.Series,
		raw_data_broker_nval: pd.DataFrame,
		raw_data_broker_sumval: pd.DataFrame,
		n_selected_cluster: int | None = None,
		training_start_index: float = 0.5,
		training_end_index: float = 0.75,
		splitted_min_n_cluster: int = 2,
		splitted_max_n_cluster: int = 5,
		stepup_n_cluster_threshold: float = 0.05,
		) -> tuple[list[str], int, float, pd.DataFrame]:

		# Delete the first self.preoffset_period_param rows from raw_data
		raw_data_close = raw_data_close.iloc[self.preoffset_period_param:]
		raw_data_broker_nval = raw_data_broker_nval.iloc[self.preoffset_period_param:,:]
		raw_data_broker_sumval = raw_data_broker_sumval.iloc[self.preoffset_period_param:,:]

		# Only get third quartile of raw_data so not over-fitting
		# length = len(raw_data_close)
		# start_index = int(length*training_start_index)
		# end_index = int(length*training_end_index)
		# raw_data_close = raw_data_close.iloc[start_index:end_index]
		# raw_data_broker_nval = raw_data_broker_nval.iloc[start_index:end_index,:]
		# raw_data_broker_sumval = raw_data_broker_sumval.iloc[start_index:end_index,:]

		if (raw_data_broker_nval == 0).all().all() or (raw_data_broker_sumval == 0).all().all():
			raise ValueError("There is no transaction for the stockcode in the selected quantile")
		
		# Cumulate value for nvol
		broker_ncum = raw_data_broker_nval.cumsum(axis=0)
		# Get correlation between raw_data_ncum and close
		broker_ncum_corr = broker_ncum.corrwith(raw_data_close,axis=0).rename("corr_ncum_close")

		# Get each broker's sum of transaction value
		broker_sumval = raw_data_broker_sumval.sum(axis=0).rename("broker_sumval")

		# Combine broker_ncum_corr and broker_sumval with date as index
		broker_features = pd.concat([broker_ncum_corr,broker_sumval],axis=1)
		# fillna
		broker_features.fillna(value=0, inplace=True)

		# Delete variable for memory management
		del raw_data_broker_nval, raw_data_broker_sumval, broker_ncum_corr, broker_sumval
		gc.collect()

		# # Obsolete Method: General Clustering
		# # Replace by separated clustering for each negative and positive correlation
		# # Standardize Features
		# broker_features_std = await self.__xy_standardize(broker_features)
		# # Clustering
		# broker_features_cluster, broker_features_centroids = await self.__kmeans_clustering(broker_features_std, "corr_ncum_close", "broker_sumval")
		
		# Standardize Features
		# Using no standarization
		broker_features_std_pos = broker_features[broker_features['corr_ncum_close']>0].copy()
		broker_features_std_neg = broker_features[broker_features['corr_ncum_close']<=0].copy()

		# Positive Clustering
		broker_features_pos, centroids_pos = await self.__kmeans_clustering(
			broker_features_std_pos, "corr_ncum_close", "broker_sumval", 
			min_n_cluster=splitted_min_n_cluster, max_n_cluster=splitted_max_n_cluster)
		# Negative Clustering
		broker_features_neg, centroids_neg = await self.__kmeans_clustering(
			broker_features_std_neg, "corr_ncum_close", "broker_sumval", 
			min_n_cluster=splitted_min_n_cluster, max_n_cluster=splitted_max_n_cluster)
		broker_features_neg["cluster"] = broker_features_neg['cluster'] + broker_features_pos['cluster'].max() + 1
		centroids_neg.index = centroids_neg.index + centroids_pos.index.max() + 1

		# Combine Positive and Negative Clustering
		broker_features_cluster = pd.concat([broker_features_pos,broker_features_neg],axis=0)
		broker_features_centroids = pd.concat([centroids_pos,centroids_neg],axis=0)

		# Get cluster label
		broker_features["cluster"] = broker_features_cluster["cluster"].astype("int")

		# Delete variable for memory management
		del broker_features_std_pos, broker_features_std_neg, \
			broker_features_pos, centroids_pos, \
			broker_features_cluster
		gc.collect()

		# Define optimum selected cluster: net transaction clusters with highest correlation to close
		selected_broker, optimum_n_selected_cluster, optimum_corr = \
			await self.__optimize_selected_cluster(
				clustered_features=broker_features,
				raw_data_close=raw_data_close,
				broker_ncum=broker_ncum,
				centroids_cluster=broker_features_centroids,
				n_selected_cluster=n_selected_cluster,
				stepup_n_cluster_threshold=stepup_n_cluster_threshold
			)

		return selected_broker, optimum_n_selected_cluster, optimum_corr, broker_features

	async def get_timeseries_cluster(self,
		df: pd.DataFrame,
		splitted_min_n_cluster: int = 4,
		splitted_max_n_cluster: int = 10
		) -> tuple[pd.DataFrame, pd.DataFrame]:
		# Scaling the dataframe for each column
		scaler = MinMaxScaler()
		df_scaled = scaler.fit_transform(df)
		df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
		
		# Dimensionality Reduction for each column
		pca = PCA(n_components=2)
		df_pca = pca.fit_transform(df_scaled.T)
		df_pca = pd.DataFrame(df_pca, columns=['PC1','PC2'], index=df_scaled.T.index)

		# KMeans Clustering
		df_cluster, centroids_cluster = await self.__kmeans_clustering(
			features=df_pca,
			x='PC1',
			y='PC2',
			min_n_cluster=splitted_min_n_cluster,
			max_n_cluster=splitted_max_n_cluster
			)
		
		return df_cluster, centroids_cluster
	
	async def __get_timeseries_cluster_corr(self,
		raw_data_close: pd.Series,
		raw_data_broker_nval: pd.DataFrame,
		df_cluster: pd.DataFrame,
		):
		# Get number of clusters from df_cluster['cluster']
		n_cluster = df_cluster['cluster'].max() + 1

		# Looping each cluster
		# Get broker_ncum for each cluster from raw_data_broker_nval
		# Get correlation between broker_ncum and raw_data_close
		# Then append to cluster_corr dataframe
		cluster_corr = pd.DataFrame()
		for i in range(n_cluster):
			brokers = df_cluster[df_cluster['cluster']==i].index
			broker_ncum = raw_data_broker_nval[brokers].sum(axis=1).cumsum(axis=0)
			broker_ncum_corr = broker_ncum.corr(raw_data_close)
			cluster_corr = pd.concat([cluster_corr, pd.DataFrame(
				{'cluster':i, 'corr_ncum_close':broker_ncum_corr}, index=[0])], axis=0)
		# fillna with 0
		cluster_corr = cluster_corr.fillna(0)
		
		# Add corr_abs column to cluster_corr
		cluster_corr['corr_ncum_close_abs'] = cluster_corr['corr_ncum_close'].abs()

		# Sort cluster_corr by correlation
		cluster_corr = cluster_corr.sort_values(by='corr_ncum_close_abs', ascending=False).reset_index(drop=True)

		return cluster_corr
	
	async def __optimize_timeseries_selected_cluster(self,
		raw_data_close: pd.Series,
		raw_data_broker_nval: pd.DataFrame,
		cluster_corr: pd.DataFrame,
		df_cluster: pd.DataFrame,
		stepup_n_cluster_threshold: float = 0.05,
		) -> tuple[list[str], int, float]:
		# Get only positive correlation from cluster_corr
		# cluster_corr = cluster_corr[cluster_corr['corr_ncum_close']>0]
		
		# Sort cluster_corr by correlation absolute value
		cluster_corr = cluster_corr.sort_values(by='corr_ncum_close_abs', ascending=False).reset_index(drop=True)

		agg_cluster_corr = [cluster_corr['corr_ncum_close_abs'].max()]
		for i in range(1,len(cluster_corr)):
			clusters = cluster_corr.iloc[:i+1,0].values
			brokers = df_cluster[df_cluster['cluster'].isin(clusters)].index
			broker_ncum = raw_data_broker_nval[brokers].sum(axis=1).cumsum(axis=0)
			broker_ncum_corr = broker_ncum.corr(raw_data_close)
			agg_cluster_corr.append(broker_ncum_corr)

		# Define optimum n_selected_cluster
		max_corr: float = np.max(agg_cluster_corr)
		index_max_corr: int = int(np.argmax(agg_cluster_corr))
		optimum_corr: float = max_corr
		optimum_n_selected_cluster: int = index_max_corr + 1

		for i in range (index_max_corr):
			if (max_corr-agg_cluster_corr[i]) < stepup_n_cluster_threshold:
				optimum_n_selected_cluster = i+1
				optimum_corr = agg_cluster_corr[i]
				break

		selected_cluster = cluster_corr.iloc[:optimum_n_selected_cluster,0].values
		selected_broker = df_cluster[df_cluster['cluster'].isin(selected_cluster)].index.to_list()

		return selected_broker, optimum_n_selected_cluster, optimum_corr

	async def __adjust_plusmin_df(self,
		df: pd.DataFrame,
		broker_cluster: pd.DataFrame,
		) -> pd.DataFrame:
		# Get brokers from broker_cluster with negative corr
		brokers = broker_cluster[broker_cluster['corr_ncum_close']<0].index.to_list()

		# Adjust plusmin_df by multiplying -1 to brokers
		df = df.copy()
		df.loc[:, brokers] *= -1

		return df
	
	async def __get_timeseries_bf_parameter(self,
		raw_data_close: pd.Series,
		raw_data_broker_nval: pd.DataFrame,
		splitted_min_n_cluster: int = 4,
		splitted_max_n_cluster: int = 10,
		training_start_index: float = 0.5,
		training_end_index: float = 0.75,
		stepup_n_cluster_threshold: float = 0.05,
		) -> tuple[list[str], int, float, pd.DataFrame, pd.DataFrame]:
		# Delete the first self.preoffset_period_param rows from raw_data
		raw_data_close = raw_data_close.iloc[self.preoffset_period_param:]
		raw_data_broker_nval = raw_data_broker_nval.iloc[self.preoffset_period_param:,:]

		# Only get third quartile of raw_data so not over-fitting
		# length = len(raw_data_close)
		# start_index = int(length*training_start_index)
		# end_index = int(length*training_end_index)
		# raw_data_close = raw_data_close.iloc[start_index:end_index]
		# raw_data_broker_nval = raw_data_broker_nval.iloc[start_index:end_index,:]

		# Get timeseries cluster
		broker_ncum = raw_data_broker_nval.cumsum(axis=0)
		df_cluster, centroids_cluster = await self.get_timeseries_cluster(
			broker_ncum, splitted_min_n_cluster, splitted_max_n_cluster)

		# Get correlation between broker_ncum each cluster and raw_data_close
		cluster_corr = await self.__get_timeseries_cluster_corr(
			raw_data_close = raw_data_close,
			raw_data_broker_nval = raw_data_broker_nval,
			df_cluster = df_cluster,
			)
		
		# Join cluster_corr to df_cluster on cluster
		df_cluster = df_cluster.join(cluster_corr.set_index('cluster'), on='cluster')

		# Adjust plusmin raw_data_broker_nval
		raw_data_broker_nval = await self.__adjust_plusmin_df(df = raw_data_broker_nval, broker_cluster = df_cluster)

		selected_broker, optimum_n_selected_cluster, optimum_corr = await self.__optimize_timeseries_selected_cluster(
			raw_data_close = raw_data_close,
			raw_data_broker_nval = raw_data_broker_nval,
			cluster_corr = cluster_corr,
			df_cluster = df_cluster,
			stepup_n_cluster_threshold=stepup_n_cluster_threshold,
			)
		
		return selected_broker, optimum_n_selected_cluster, optimum_corr, df_cluster, broker_ncum

	async def calc_wf_indicators(self,
		raw_data_full: pd.DataFrame,
		raw_data_broker_nvol: pd.DataFrame,
		raw_data_broker_nval: pd.DataFrame,
		raw_data_broker_sumval: pd.DataFrame,
		selected_broker: list[str],
		period_mf: int = 1,
		period_prop: int = 10,
		period_pricecorrel: int = 10,
		period_mapricecorrel: int = 100,
		period_vwap:int = 21,
		pow_high_prop: int = 40,
		pow_high_pricecorrel: int = 50,
		pow_high_mapricecorrel: int = 30,
		pow_medium_prop: int = 20,
		pow_medium_pricecorrel: int = 30,
		pow_medium_mapricecorrel: int = 30,
		preoffset_period_param: int = 50,
		) -> pd.DataFrame:
		# OHLC
		# raw_data_full

		# Broker Data Prep
		selected_data_broker_nvol = raw_data_broker_nvol[selected_broker].sum(axis=1)
		selected_data_broker_nval = raw_data_broker_nval[selected_broker].sum(axis=1)
		selected_data_broker_sumval = raw_data_broker_sumval[selected_broker].sum(axis=1)

		# Net Value for Volume Profile
		raw_data_full["netval"] = selected_data_broker_nval
		
		# Whale Volume Flow
		raw_data_full["valflow"] = selected_data_broker_nval.cumsum()

		# Whale Money Flow
		raw_data_full['mf'] = selected_data_broker_nval.rolling(window=period_mf).sum()

		# Whale Proportion
		raw_data_full['prop'] = selected_data_broker_sumval.rolling(window=period_prop).sum().abs() \
			/ (raw_data_full['value'].rolling(window=period_prop).sum()*2)

		# Whale Net Proportion
		raw_data_full['netprop'] = selected_data_broker_nval.rolling(window=period_prop).sum().abs() \
			/ (raw_data_full['value'].rolling(window=period_prop).sum()*2)

		# Whale correlation
		raw_data_full['pricecorrel'] = raw_data_full["valflow"].rolling(window=period_pricecorrel).corr(raw_data_full['close'])

		# Whale MA correlation
		raw_data_full['mapricecorrel'] = raw_data_full['pricecorrel'].rolling(window=period_mapricecorrel).mean()

		# Whale-VWAP
		raw_data_full['vwap'] = (selected_data_broker_nval.rolling(window=period_vwap).apply(lambda x: x[x>0].sum()))\
			/(selected_data_broker_nvol.rolling(window=period_vwap).apply(lambda x: x[x>0].sum()))
		raw_data_full['vwap'] = raw_data_full['vwap'].mask(raw_data_full['vwap'].le(0)).ffill()

		# Whale Power
		raw_data_full['pow'] = \
			np.where(
				(raw_data_full["prop"]>(pow_high_prop/100)) & \
				(raw_data_full['pricecorrel']>(pow_high_pricecorrel/100)) & \
				(raw_data_full['mapricecorrel']>(pow_high_mapricecorrel/100)), \
				3,
				np.where(
					(raw_data_full["prop"]>(pow_medium_prop/100)) & \
					(raw_data_full['pricecorrel']>(pow_medium_pricecorrel/100)) & \
					(raw_data_full['mapricecorrel']>(pow_medium_mapricecorrel/100)), \
					2, \
					1
				)
			)

		# End of Method: Return Processed Raw Data to BF Indicators
		return raw_data_full.loc[self.startdate:self.enddate]

	async def chart(self,media_type: str | None = None):
		assert self.stockcode is not None
		fig = await genchart.quantist_stock_chart(
			stockcode=self.stockcode,
			wf_indicators=self.wf_indicators,
			analysis_method=dp.AnalysisMethod.broker,
			period_prop=self.period_prop,
			period_pricecorrel=self.period_pricecorrel,
			period_mapricecorrel=self.period_mapricecorrel,
			period_vwap=self.period_vwap,
			selected_broker=self.selected_broker,
			optimum_n_selected_cluster=self.optimum_n_selected_cluster,
			optimum_corr=self.optimum_corr,
			)
		if media_type in ["png","jpeg","jpg","webp","svg"]:
			return await genchart.fig_to_image(fig,media_type)
		elif media_type == "json":
			return await genchart.fig_to_json(fig)
		else:
			return fig
	
	async def broker_cluster_chart(self,media_type: str | None = None):
		assert self.stockcode is not None

		if self.clustering_method == dp.ClusteringMethod.timeseries:
			scaler = MinMaxScaler()
			broker_ncum_scaled = scaler.fit_transform(self.broker_ncum)
			broker_ncum_scaled = pd.DataFrame(broker_ncum_scaled,
					columns=self.broker_ncum.columns, index=self.broker_ncum.index)
			
			fig = await genchart.broker_cluster_timeseries_chart(
				broker_cluster=self.broker_cluster,
				broker_ncum=broker_ncum_scaled,
				raw_data_close=self.wf_indicators["close"],
				code=self.stockcode,
			)
		else:
			fig = await genchart.broker_cluster_chart(
				broker_features=self.broker_features,
				code=self.stockcode,
			)
		if media_type in ["png","jpeg","jpg","webp","svg"]:
			return await genchart.fig_to_image(fig,media_type)
		elif media_type == "json":
			return await genchart.fig_to_json(fig)
		else:
			return fig

class WhaleRadar():
	def __init__(self,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		y_axis_type: dp.ListRadarType = dp.ListRadarType.correlation,
		stockcode_excludes: set[str] = set(),
		include_composite: bool = False,
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		n_selected_cluster:int | None = None,
		radar_period: int | None = None,
		period_mf: int | None = None,
		period_pricecorrel: int | None = None,
		default_months_range: int | None = None,
		training_start_index: float | None = None,
		training_end_index: float | None = None,
		min_n_cluster: int | None = None,
		max_n_cluster: int | None = None,
		splitted_min_n_cluster: int | None = None,
		splitted_max_n_cluster: int | None = None,
		stepup_n_cluster_threshold: int | None = None,
		filter_opt_corr: float | None = None,
		dbs: db.Session = next(db.get_dbs()),
		) -> None:
		self.startdate = startdate
		self.enddate = enddate
		self.y_axis_type = y_axis_type
		self.stockcode_excludes = stockcode_excludes
		self.include_composite = include_composite
		self.screener_min_value = screener_min_value
		self.screener_min_frequency = screener_min_frequency
		self.n_selected_cluster = n_selected_cluster
		self.radar_period = radar_period
		self.period_mf = period_mf
		self.period_pricecorrel = period_pricecorrel
		self.default_months_range = default_months_range
		self.training_start_index = training_start_index
		self.training_end_index = training_end_index
		self.min_n_cluster = min_n_cluster
		self.max_n_cluster = max_n_cluster
		self.splitted_min_n_cluster = splitted_min_n_cluster
		self.splitted_max_n_cluster = splitted_max_n_cluster
		self.stepup_n_cluster_threshold = stepup_n_cluster_threshold
		self.filter_opt_corr = filter_opt_corr
		self.dbs = dbs

		self.radar_indicators:pd.DataFrame
	
	async def fit (self) -> WhaleRadar:
		# Get default bf params
		await self._get_default_radar(dbs=self.dbs)
		assert self.screener_min_value is not None
		assert self.screener_min_frequency is not None
		assert self.default_months_range is not None
		assert self.training_end_index is not None
		assert self.training_start_index is not None
		assert self.splitted_min_n_cluster is not None
		assert self.splitted_max_n_cluster is not None
		assert self.filter_opt_corr is not None

		# Get Filtered StockCodes
		self.filtered_stockcodes = await self._get_stockcodes(
			screener_min_value=self.screener_min_value,
			screener_min_frequency=self.screener_min_frequency,
			stockcode_excludes=self.stockcode_excludes,
			dbs=self.dbs)

		# Get raw data
		raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumvol, raw_data_broker_sumval, self.filtered_stockcodes = \
			await self._get_stock_raw_data(
				filtered_stockcodes=self.filtered_stockcodes,
				startdate=self.startdate,
				enddate=self.enddate,
				default_months_range=self.default_months_range,
				dbs=self.dbs
				)

		# Get broker flow parameters for each stock in filtered_stockcodes
		self.selected_broker, self.optimum_n_selected_cluster, self.optimum_corr, self.broker_features = \
			await self._get_bf_parameters(
				raw_data_close=raw_data_full["close"],
				raw_data_broker_nval=raw_data_broker_nval,
				raw_data_broker_sumval=raw_data_broker_sumval,
				n_selected_cluster=self.n_selected_cluster,
				training_start_index=self.training_start_index,
				training_end_index=self.training_end_index,
				splitted_min_n_cluster=self.splitted_min_n_cluster,
				splitted_max_n_cluster=self.splitted_max_n_cluster,
			)
		
		# Filter code based on self.optimum_corr should be greater than self.filter_opt_corr
		self.filtered_stockcodes, raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval = \
			await self._get_filtered_stockcodes_by_corr(
				filter_opt_corr=self.filter_opt_corr,
				optimum_corr=self.optimum_corr,
				filtered_stockcodes=self.filtered_stockcodes,
				raw_data_full=raw_data_full,
				raw_data_broker_nvol=raw_data_broker_nvol,
				raw_data_broker_nval=raw_data_broker_nval,
				raw_data_broker_sumval=raw_data_broker_sumval
			)

		# Get radar period filtered stockdata
		self.startdate, self.enddate, raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval = \
			await self._get_radar_period_filtered_stock_data(
				startdate = self.startdate,
				radar_period = self.radar_period,
				period_predata=self.period_pricecorrel,
				raw_data_full = raw_data_full,
				raw_data_broker_nvol = raw_data_broker_nvol,
				raw_data_broker_nval = raw_data_broker_nval,
				raw_data_broker_sumval = raw_data_broker_sumval,
			)
		
		# Get sum of selected broker transaction for each stock
		selected_broker_nvol, selected_broker_nval, selected_broker_sumval = \
			await self._sum_selected_broker_transaction(
				raw_data_broker_nvol=raw_data_broker_nvol,
				raw_data_broker_nval=raw_data_broker_nval,
				raw_data_broker_sumval=raw_data_broker_sumval,
				selected_broker=self.selected_broker,
			)
		
		# Get Whale Radar Indicators
		self.radar_indicators = await self._calc_radar_indicators(
			raw_data_full = raw_data_full,
			selected_broker_nval = selected_broker_nval,
			y_axis_type=self.y_axis_type,
			)
		
		return self
	
	async def _get_default_radar(self, dbs:db.Session = next(db.get_dbs())) -> pd.Series:
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter(
				(db.DataParam.param.like("default_months_range")) | \
				(db.DataParam.param.like("default_radar_%")) | \
				(db.DataParam.param.like("default_screener_%")) | \
				(db.DataParam.param.like("default_bf_%"))
				)
		default_radar = pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])

		# Data Parameter
		self.training_start_index = (int(default_radar['default_bf_training_start_index'])-50)/(100/2) if self.training_start_index is None else self.training_start_index/100
		self.training_end_index = (int(default_radar['default_bf_training_end_index'])-50)/(100/2) if self.training_end_index is None else self.training_end_index/100
		self.min_n_cluster = int(default_radar['default_bf_min_n_cluster']) if self.min_n_cluster is None else self.min_n_cluster
		self.max_n_cluster = int(default_radar['default_bf_max_n_cluster']) if self.max_n_cluster is None else self.max_n_cluster
		self.splitted_min_n_cluster = int(default_radar['default_bf_splitted_min_n_cluster']) if self.splitted_min_n_cluster is None else self.splitted_min_n_cluster
		self.splitted_max_n_cluster = int(default_radar['default_bf_splitted_max_n_cluster']) if self.splitted_max_n_cluster is None else self.splitted_max_n_cluster
		self.stepup_n_cluster_threshold = int(default_radar['default_bf_stepup_n_cluster_threshold'])/100 if self.stepup_n_cluster_threshold is None else self.stepup_n_cluster_threshold/100
		
		self.radar_period = int(default_radar['default_radar_period']) if self.radar_period is None else self.radar_period
		self.screener_min_value = int(default_radar['default_screener_min_value']) if self.screener_min_value is None else self.screener_min_value
		self.screener_min_frequency = int(default_radar['default_screener_min_frequency']) if self.screener_min_frequency is None else self.screener_min_frequency
		self.filter_opt_corr = int(default_radar['default_radar_filter_opt_corr'])/100 if self.filter_opt_corr is None else self.filter_opt_corr/100
		
		self.default_months_range = int((int(default_radar['default_months_range'])/2) + int(self.radar_period/20)) if self.startdate is None else self.default_months_range
		
		return default_radar
	
	async def _get_stockcodes(self,
		screener_min_value: int = 5000000000,
		screener_min_frequency: int = 1000,
		stockcode_excludes: set[str] = set(),
		dbs: db.Session = next(db.get_dbs())
		) -> pd.Series:
		"""
		Get filtered stockcodes
		Filtered by:value>screener_min_value, 
					frequency>screener_min_frequency 
					stockcode_excludes
		"""
		# Query Definition
		stockcode_excludes_lower = set(x.lower() for x in stockcode_excludes) if stockcode_excludes is not None else set()
		qry = dbs.query(db.ListStock.code)\
			.filter((db.ListStock.value > screener_min_value) &
					(db.ListStock.frequency > screener_min_frequency) &
					(db.ListStock.code.not_in(stockcode_excludes_lower)))
		
		# Query Fetching: filtered_stockcodes
		return pd.Series(pd.read_sql(sql=qry.statement,con=dbs.bind).reset_index(drop=True)['code'])
	
	# Get Net Val Sum Val Broker Transaction
	async def __get_nvsv_broker_transaction(self,
		raw_data_broker_full: pd.DataFrame
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		raw_data_broker_nvol = raw_data_broker_full.pivot(index=None,columns="broker",values="nvol")
		raw_data_broker_nval = raw_data_broker_full.pivot(index=None,columns="broker",values="nval")
		raw_data_broker_sumvol = raw_data_broker_full.pivot(index=None,columns="broker",values="sumvol")
		raw_data_broker_sumval = raw_data_broker_full.pivot(index=None,columns="broker",values="sumval")

		# Fill na
		raw_data_broker_nvol.fillna(value=0, inplace=True)
		raw_data_broker_nval.fillna(value=0, inplace=True)
		raw_data_broker_sumvol.fillna(value=0, inplace=True)
		raw_data_broker_sumval.fillna(value=0, inplace=True)

		# Return
		return raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumvol, raw_data_broker_sumval

	async def __get_full_broker_transaction(self,
		filtered_stockcodes: pd.Series = ...,
		enddate: datetime.date = datetime.date.today(),
		default_months_range: int = 12,
		dbs: db.Session = next(db.get_dbs()),
		) -> pd.DataFrame:

		start_date = enddate - relativedelta(months=default_months_range)

		# Query Definition
		qry = dbs.query(
			db.StockTransaction.date,
			db.StockTransaction.code,
			db.StockTransaction.broker,
			db.StockTransaction.bvol,
			db.StockTransaction.svol,
			db.StockTransaction.bval,
			db.StockTransaction.sval,
			(db.StockTransaction.bvol - db.StockTransaction.svol).label("nvol"), # type: ignore
			(db.StockTransaction.bval - db.StockTransaction.sval).label("nval"), # type: ignore
			(db.StockTransaction.bvol + db.StockTransaction.svol).label("sumvol"), # type: ignore
			(db.StockTransaction.bval + db.StockTransaction.sval).label("sumval") # type: ignore
		).filter(db.StockTransaction.code.in_(filtered_stockcodes.to_list()))\
		.filter(db.StockTransaction.date.between(start_date, enddate))\
		.order_by(db.StockTransaction.code.asc(), db.StockTransaction.date.asc(), db.StockTransaction.broker.asc())

		# Main Query Fetching
		raw_data_broker_full = pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=["date"])\
			.reset_index(drop=True).set_index(["code","date"])

		# Data Cleansing: fillna
		raw_data_broker_full.fillna(value=0, inplace=True)

		return raw_data_broker_full
	
	async def __get_stock_price_data(self,
		filtered_stockcodes: pd.Series = ...,
		startdate:datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		default_months_range: int = 12,
		minimum_training_set: int = 0,
		dbs: db.Session = next(db.get_dbs()),
		) -> pd.DataFrame:

		# Check data availability if startdate is not None
		if startdate is not None:
			qry = dbs.query(db.StockData.code
				).filter(db.StockData.code.in_(filtered_stockcodes.to_list())
				).filter(db.StockData.date.between(startdate, enddate)
				).group_by(db.StockData.code)
			
			# Query Fetching
			raw_data = pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=["date"])

			# Check how many row is returned
			if raw_data.shape[0] == 0:
				raise ValueError("No data available inside date range")

		start_date = enddate - relativedelta(months=default_months_range)

		# Query Definition
		# Filter only data that has data in date range more than minimum_training_set rows
		sub_qry = dbs.query(db.StockData.code, func.count(db.StockData.code).label("count"))\
			.filter(db.StockData.code.in_(filtered_stockcodes.to_list()))\
			.filter(db.StockData.date.between(start_date, enddate))\
			.group_by(db.StockData.code)\
			.having(func.count(db.StockData.code) > minimum_training_set)\
			.subquery()

		qry = dbs.query(db.StockData.code,db.StockData.date,db.StockData.close,db.StockData.value)\
			.join(sub_qry, db.StockData.code == sub_qry.c.code)\
			.filter(db.StockData.code.in_(filtered_stockcodes.to_list()))\
			.filter(db.StockData.date.between(start_date, enddate))\
			.order_by(db.StockData.code.asc(), db.StockData.date.asc())

		# Main Query Fetching
		raw_data_full = pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=["date"])\
			.reset_index(drop=True).set_index(["code","date"])

		# End of Method: Return or Assign Attribute
		return raw_data_full

	async def _get_stock_raw_data(self,
		filtered_stockcodes: pd.Series = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_months_range: int = 6,
		dbs: db.Session = next(db.get_dbs()),
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
		MINIMUM_TRAINING_SET: int = 5

		# Get Stockdata Full
		raw_data_full = await self.__get_stock_price_data(
			filtered_stockcodes=filtered_stockcodes,
			startdate=startdate,
			enddate=enddate,
			default_months_range=default_months_range,
			minimum_training_set=MINIMUM_TRAINING_SET,
			dbs=dbs
			)

		# Get filtered_stockcodes from raw_data_full first level
		filtered_stockcodes = raw_data_full.index.get_level_values(0).unique().to_series()

		# Get Raw Data Broker Full
		raw_data_broker_full = await self.__get_full_broker_transaction(
			filtered_stockcodes=filtered_stockcodes,
			enddate=enddate,
			default_months_range=default_months_range,
			dbs=dbs
			)
		
		# Transform Raw Data Broker
		raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumvol, raw_data_broker_sumval = \
			await self.__get_nvsv_broker_transaction(raw_data_broker_full=raw_data_broker_full)

		return raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumvol, raw_data_broker_sumval, filtered_stockcodes
	
	async def __get_selected_broker(self,
		clustered_features: pd.DataFrame,
		centroids_cluster: pd.DataFrame,
		n_selected_cluster: int = 1,
		) -> list[str]:

		# Get index of max value in column 0 in centroid
		selected_cluster = (centroids_cluster[0]).nlargest(n_selected_cluster).index.tolist()
		# selected_cluster = (abs(centroids_cluster[0]*centroids_cluster[1])).nlargest(n_selected_cluster).index.tolist()
		
		# Get sorted selected broker
		selected_broker = clustered_features.loc[clustered_features["cluster"].isin(selected_cluster), :]\
			.sort_values(by="corr_ncum_close", ascending=False)\
			.index.tolist()

		return selected_broker

	async def __get_corr_selected_broker_ncum(self,
		clustered_features: pd.DataFrame,
		raw_data_close: pd.Series,
		broker_ncum: pd.DataFrame,
		centroids_cluster: pd.DataFrame,
		n_selected_cluster: int = 1,
		) -> float:
		selected_broker = await self.__get_selected_broker(
			clustered_features=clustered_features,
			centroids_cluster=centroids_cluster,
			n_selected_cluster=n_selected_cluster
			)

		# Get selected broker transaction by columns of net_stockdatatransaction, then sum each column to aggregate to date
		selected_broker_ncum = broker_ncum[selected_broker].sum(axis=1)

		# Return correlation between close and selected_broker_ncum
		return selected_broker_ncum.corr(raw_data_close)

	async def __optimize_selected_cluster(self,
		clustered_features: pd.DataFrame,
		raw_data_close: pd.Series,
		broker_ncum: pd.DataFrame,
		centroids_cluster: pd.DataFrame,
		stepup_n_cluster_threshold: float = 0.05,
		n_selected_cluster: int | None = None,
		) -> tuple[list[str], int, float]:
		# Check does n_selected_cluster already defined
		if n_selected_cluster is None:
			# Define correlation param
			corr_list = []

			# Iterate optimum n_cluster
			for n_selected_cluster in range(1,len(centroids_cluster)):
				# Get correlation between close and selected_broker_ncum
				selected_broker_ncum_corr = await self.__get_corr_selected_broker_ncum(
					clustered_features,
					raw_data_close,
					broker_ncum,
					centroids_cluster,
					n_selected_cluster
					)
				# Get correlation
				corr_list.append(selected_broker_ncum_corr)

			# Define optimum n_selected_cluster
			max_corr: float = np.max(corr_list)
			index_max_corr: int = int(np.argmax(corr_list))
			optimum_corr: float = max_corr
			optimum_n_selected_cluster: int = index_max_corr + 1

			for i in range (index_max_corr):
				if (max_corr-corr_list[i]) < stepup_n_cluster_threshold:
					optimum_n_selected_cluster = i+1
					optimum_corr = corr_list[i]
					break
		# -- End of if

		# If n_selected_cluster is defined
		else:
			optimum_n_selected_cluster: int = n_selected_cluster
			optimum_corr = await self.__get_corr_selected_broker_ncum(
				clustered_features, 
				raw_data_close, 
				broker_ncum, 
				centroids_cluster, 
				n_selected_cluster
				)

		# Get Selected Broker from optimum n_selected_cluster
		selected_broker = await self.__get_selected_broker(
			clustered_features=clustered_features,
			centroids_cluster=centroids_cluster,
			n_selected_cluster=optimum_n_selected_cluster
			)

		return selected_broker, optimum_n_selected_cluster, optimum_corr

	async def __kmeans_clustering(self,
		features: pd.DataFrame,
		x: str,
		y: str,
		min_n_cluster:int = 4,
		max_n_cluster:int = 10,
		) -> tuple[pd.DataFrame, pd.DataFrame]:

		# Get X and Y
		X = features[[x,y]].values
		# Define silhouette param
		silhouette_coefficient = []
		max_n_cluster = min(max_n_cluster, len(X)-1)

		# Iterate optimum n_cluster
		for n_cluster in range(min_n_cluster, max_n_cluster+1):
			# Clustering
			kmeans = KMeans(init="k-means++", n_init='auto', n_clusters=n_cluster, random_state=0).fit(X)
			score = silhouette_score(X, kmeans.labels_)
			silhouette_coefficient.append(score)
		# Define optimum n_cluster
		optimum_n_cluster = int(np.argmax(silhouette_coefficient)) + min_n_cluster

		# Clustering with optimum n cluster
		kmeans = KMeans(init="k-means++", n_init='auto', n_clusters=optimum_n_cluster, random_state=0).fit(X)
		# Get cluster label
		features["cluster"] = kmeans.labels_
		# Get location of cluster center
		centroids_cluster = pd.DataFrame(kmeans.cluster_centers_)

		return features, centroids_cluster

	async def _get_bf_parameters(self,
		raw_data_close: pd.Series,
		raw_data_broker_nval: pd.DataFrame,
		raw_data_broker_sumval: pd.DataFrame,
		n_selected_cluster: int | None = None,
		training_start_index: float = 0.5,
		training_end_index: float = 0.75,
		splitted_min_n_cluster: int = 2,
		splitted_max_n_cluster: int = 5,
		stepup_n_cluster_threshold: float = 0.05,
		) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

		# Only get third quartile of raw_data so not over-fitting
		# length = raw_data_close.groupby(by='code').size()
		# start_index = (length*training_start_index).astype('int')
		# end_index = (length*training_end_index).astype('int')
		# raw_data_close = raw_data_close.groupby(by='code', group_keys=False)\
		# 	.apply(lambda x: x.iloc[start_index.loc[x.name]:end_index.loc[x.name]])
		# raw_data_broker_nval = raw_data_broker_nval.groupby(by='code', group_keys=False)\
		# 	.apply(lambda x: x.iloc[start_index.loc[x.name]:end_index.loc[x.name]])
		
		# Only get raw_data_broker_nval groupby level code that doesn' all zero
		nval_true = raw_data_broker_nval.groupby(by='code', group_keys=False)\
			.apply(lambda x: (x!=0).any().any())
		sumval_true = raw_data_broker_sumval.groupby(by='code', group_keys=False)\
			.apply(lambda x: (x!=0).any().any())
		transaction_true = nval_true & sumval_true
		raw_data_close = raw_data_close.loc[transaction_true.index[transaction_true]]
		raw_data_broker_nval = raw_data_broker_nval.loc[transaction_true.index[transaction_true]]
		raw_data_broker_sumval = raw_data_broker_sumval.loc[transaction_true.index[transaction_true]]

		# Cumulate volume for nvol
		broker_ncum = raw_data_broker_nval.groupby(by='code').cumsum(axis=0)
		# Get correlation between raw_data_ncum and close
		corr_ncum_close = broker_ncum.groupby(by='code').corrwith(raw_data_close,axis=0) # type: ignore

		# Get each broker's sum of transaction value
		broker_sumval = raw_data_broker_sumval.groupby(by='code').sum()

		# fillna
		corr_ncum_close.fillna(value=0, inplace=True)
		broker_sumval.fillna(value=0, inplace=True)
		
		# Create broker features from corr_ncum_close and broker_sumval
		corr_ncum_close = corr_ncum_close.unstack().swaplevel(0,1).sort_index(level=0).rename('corr_ncum_close') # type: ignore
		broker_sumval = broker_sumval.unstack().swaplevel(0,1).sort_index(level=0).rename('broker_sumval') # type: ignore
		broker_features = pd.concat([corr_ncum_close, broker_sumval], axis=1)

		# Delete variable for memory management
		del raw_data_broker_nval, raw_data_broker_sumval, corr_ncum_close, broker_sumval
		gc.collect()

		# Standardize Features
		# Using no standarization
		# Get the column name from corr_ncum_close for each index that has correlation > 0
		broker_features_std_pos = broker_features[broker_features['corr_ncum_close']>0]
		broker_features_std_neg = broker_features[broker_features['corr_ncum_close']<=0]
		
		# Positive Clustering
		broker_features_pos = pd.DataFrame()
		centroids_pos = pd.DataFrame()
		for code in broker_features_std_pos.index.get_level_values('code').unique():
			features, centroids = \
				await self.__kmeans_clustering(
					features=broker_features_std_pos.loc[code,:],
					x='corr_ncum_close',
					y='broker_sumval',
					min_n_cluster=splitted_min_n_cluster,
					max_n_cluster=splitted_max_n_cluster,
				)
			features['code'] = code
			features = features.set_index('code', append=True).swaplevel(0,1).sort_index(level=0)
			broker_features_pos = pd.concat([broker_features_pos, features], axis=0)
			centroids['code'] = code
			centroids = centroids.set_index('code', append=True).swaplevel(0,1).sort_index(level=0)
			centroids_pos = pd.concat([centroids_pos, centroids], axis=0)

		# Negative Clustering
		broker_features_neg = pd.DataFrame()
		centroids_neg = pd.DataFrame()
		for code in broker_features_std_neg.index.get_level_values('code').unique():
			features, centroids = \
				await self.__kmeans_clustering(
					features=broker_features_std_neg.loc[code,:],
					x='corr_ncum_close',
					y='broker_sumval',
					min_n_cluster=splitted_min_n_cluster,
					max_n_cluster=splitted_max_n_cluster,
				)
			features['code'] = code
			features = features.set_index('code', append=True).swaplevel(0,1).sort_index(level=0)
			if code in broker_features_pos.index.get_level_values('code'):
				features = features + (broker_features_pos.loc[(code),"cluster"].max()) + 1 # type: ignore
				broker_features_neg = pd.concat([broker_features_neg, features], axis=0)

			centroids['code'] = code
			if code in broker_features_pos.index.get_level_values('code'):
				centroids.index = centroids.index + centroids_pos.loc[code,:].index.max() + 1
			centroids = centroids.set_index('code', append=True).swaplevel(0,1).sort_index(level=0)
			centroids_neg = pd.concat([centroids_neg, centroids], axis=0)
			
		# Combine Positive and Negative Clustering
		broker_features_cluster = pd.concat([broker_features_pos,broker_features_neg],axis=0)
		broker_features_centroids = pd.concat([centroids_pos,centroids_neg],axis=0)

		# Get cluster label
		broker_features["cluster"] = broker_features_cluster["cluster"].astype("int")

		# Delete variable for memory management
		del broker_features_std_pos, broker_features_std_neg, \
			broker_features_pos, centroids_pos, \
			broker_features_neg, centroids_neg, \
			broker_features_cluster
		gc.collect()

		# Define optimum selected cluster: net transaction clusters with highest correlation to close
		selected_broker = {}
		optimum_n_selected_cluster = {}
		optimum_corr = {}
		for code in broker_features.index.get_level_values('code').unique():
			assert isinstance(code, str)
			selected_broker_code, optimum_n_selected_cluster_code, optimum_corr_code = \
				await self.__optimize_selected_cluster(
					clustered_features=broker_features.loc[code,:],
					raw_data_close=raw_data_close.loc[code],
					broker_ncum=broker_ncum.loc[code,:],
					centroids_cluster=broker_features_centroids.loc[code,:],
					n_selected_cluster=n_selected_cluster,
					stepup_n_cluster_threshold=stepup_n_cluster_threshold
				)
			selected_broker[code] = selected_broker_code
			optimum_n_selected_cluster[code] = optimum_n_selected_cluster_code
			optimum_corr[code] = optimum_corr_code
		
		optimum_n_selected_cluster = pd.DataFrame.from_dict(optimum_n_selected_cluster, orient='index').rename(columns={0:'optimum_n_selected_cluster'})
		optimum_corr = pd.DataFrame.from_dict(optimum_corr, orient='index').rename(columns={0:'optimum_corr'})

		return selected_broker, optimum_n_selected_cluster, optimum_corr, broker_features

	async def _sum_selected_broker_transaction(self,
		raw_data_broker_nvol: pd.DataFrame,
		raw_data_broker_nval: pd.DataFrame,
		raw_data_broker_sumval: pd.DataFrame,
		selected_broker: dict,
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		
		selected_broker_nvol = pd.DataFrame(raw_data_broker_nvol.groupby(level="code").apply(lambda x: x.loc[:,selected_broker[x.name]].sum(axis=1)).reset_index(level=0, drop=True)).rename(columns={0:'broker_nvol'})
		selected_broker_nval = pd.DataFrame(raw_data_broker_nval.groupby(level="code").apply(lambda x: x.loc[:,selected_broker[x.name]].sum(axis=1)).reset_index(level=0, drop=True)).rename(columns={0:'broker_nval'})
		selected_broker_sumval = pd.DataFrame(raw_data_broker_sumval.groupby(level="code").apply(lambda x: x.loc[:,selected_broker[x.name]].sum(axis=1)).reset_index(level=0, drop=True)).rename(columns={0:'broker_sumval'})
		
		return selected_broker_nvol, selected_broker_nval, selected_broker_sumval
	
	async def _get_filtered_stockcodes_by_corr(self,
		filter_opt_corr: float,
		optimum_corr: pd.DataFrame,
		filtered_stockcodes: pd.Series,
		raw_data_full: pd.DataFrame,
		raw_data_broker_nvol: pd.DataFrame,
		raw_data_broker_nval: pd.DataFrame,
		raw_data_broker_sumval: pd.DataFrame,
		) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		# Filter code based on self.optimum_corr should be greater than filter_opt_corr
		filtered_stockcodes = \
			filtered_stockcodes[(abs(optimum_corr['optimum_corr']) > filter_opt_corr)]\
			.reset_index(drop=True)
		raw_data_full = \
			raw_data_full[raw_data_full.index.get_level_values(0).isin(filtered_stockcodes)]
		raw_data_broker_nvol = \
			raw_data_broker_nvol[raw_data_broker_nvol.index.get_level_values(0).isin(filtered_stockcodes)]
		raw_data_broker_nval = \
			raw_data_broker_nval[raw_data_broker_nval.index.get_level_values(0).isin(filtered_stockcodes)]
		raw_data_broker_sumval = \
			raw_data_broker_sumval[raw_data_broker_sumval.index.get_level_values(0).isin(filtered_stockcodes)]

		return filtered_stockcodes, raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval

	async def _get_radar_period_filtered_stock_data(self,
		startdate: datetime.date | None = None,
		radar_period: int | None = None,
		period_predata: int | None = 0,
		raw_data_full: pd.DataFrame = ...,
		raw_data_broker_nvol: pd.DataFrame = ...,
		raw_data_broker_nval: pd.DataFrame = ...,
		raw_data_broker_sumval: pd.DataFrame = ...,
		) -> tuple[datetime.date, datetime.date, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		# Update startdate, and enddate based on Data Queried
		# Then update raw_data_full, raw_data_broker_nvol, raw_data_broker_nval
		enddate: datetime.date = raw_data_full.index.get_level_values("date").date.max() # type: ignore
		if startdate is None:
			startdate = raw_data_full.groupby("code").tail(radar_period).index.get_level_values("date").date.min() # type: ignore
		assert startdate is not None

		# Choose the maximum data length between radar_period, period_predata, and startdate
		radar_period = radar_period if radar_period is not None else 0
		period_predata = period_predata if period_predata is not None else 0
		bar_range = max(radar_period, period_predata, raw_data_full.query("date >= @startdate").groupby('code').size().max())

		# Get only bar_range rows from last row for each group by code from raw_data_full
		raw_data_full = raw_data_full.groupby("code").tail(bar_range)
		raw_data_broker_nvol = raw_data_broker_nvol.groupby("code").tail(bar_range)
		raw_data_broker_nval = raw_data_broker_nval.groupby("code").tail(bar_range)
		raw_data_broker_sumval = raw_data_broker_sumval.groupby("code").tail(bar_range)
		
		assert startdate is not None
		return startdate, enddate, raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval

	async def _calc_radar_indicators(self,
		raw_data_full: pd.DataFrame,
		selected_broker_nval: pd.DataFrame,
		y_axis_type: dp.ListRadarType = dp.ListRadarType.correlation,
		) -> pd.DataFrame:

		radar_indicators = pd.DataFrame()
		
		# Y Axis: WMF
		radar_indicators["mf"] = selected_broker_nval.groupby("code").sum()

		# X Axis:
		if y_axis_type == dp.ListRadarType.correlation:
			selected_broker_nval_cumsum = selected_broker_nval.groupby(level='code').cumsum()
			radar_indicators[y_axis_type.value] = selected_broker_nval_cumsum.groupby('code')\
				.corrwith(raw_data_full['close'],axis=0) # type: ignore
		elif y_axis_type == dp.ListRadarType.changepercentage:
			radar_indicators[y_axis_type.value] = \
				(raw_data_full.groupby('code')['close'].nth([-1]) \
				-raw_data_full.groupby('code')['close'].nth([0])) \
				/raw_data_full.groupby('code')['close'].nth([0])
		else:
			raise ValueError("Not a valid radar type")

		return radar_indicators

	async def chart(self, media_type: dp.ListMediaType | None = None):
		assert self.startdate is not None

		fig = await genchart.radar_chart(
			startdate=self.startdate,
			enddate=self.enddate,
			y_axis_type=self.y_axis_type,
			method="Whale",
			radar_indicators=self.radar_indicators,
		)

		if media_type in [
			dp.ListMediaType.png,
			dp.ListMediaType.jpeg,
			dp.ListMediaType.jpg,
			dp.ListMediaType.webp,
			dp.ListMediaType.svg
		]:
			return await genchart.fig_to_image(fig,media_type)
		elif media_type == dp.ListMediaType.json:
			return await genchart.fig_to_json(fig)
		else:
			return fig

class ScreenerBase(WhaleRadar):
	def __init__(self,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		stockcode_excludes: set[str] = set(),
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		n_selected_cluster:int | None = None,
		radar_period: int | None = None,
		period_mf: int | None = None,
		period_pricecorrel: int | None = None,
		default_months_range: int | None = None,
		training_start_index: float | None = None,
		training_end_index: float | None = None,
		min_n_cluster: int | None = None,
		max_n_cluster: int | None = None,
		splitted_min_n_cluster: int | None = None,
		splitted_max_n_cluster: int | None = None,
		stepup_n_cluster_threshold: int | None = None,
		filter_opt_corr: int | None = None,
		dbs: db.Session = next(db.get_dbs()),
		) -> None:
		super().__init__(
			startdate=startdate,
			enddate=enddate,
			stockcode_excludes=stockcode_excludes,
			screener_min_value=screener_min_value,
			screener_min_frequency=screener_min_frequency,
			n_selected_cluster=n_selected_cluster,
			radar_period=radar_period,
			period_mf=period_mf,
			period_pricecorrel=period_pricecorrel,
			default_months_range=default_months_range,
			training_start_index=training_start_index,
			training_end_index=training_end_index,
			min_n_cluster=min_n_cluster,
			max_n_cluster=max_n_cluster,
			splitted_min_n_cluster=splitted_min_n_cluster,
			splitted_max_n_cluster=splitted_max_n_cluster,
			stepup_n_cluster_threshold=stepup_n_cluster_threshold,
			filter_opt_corr=filter_opt_corr,
			dbs=dbs,
		)
	
	async def _fit_base(self, predata: str | None = None) -> ScreenerBase:
		# Get default bf params
		default_radar = await super()._get_default_radar(dbs=self.dbs)
		assert self.radar_period is not None
		assert self.screener_min_value is not None
		assert self.screener_min_frequency is not None
		assert self.default_months_range is not None
		assert self.training_end_index is not None
		assert self.training_start_index is not None
		assert self.splitted_min_n_cluster is not None
		assert self.splitted_max_n_cluster is not None
		assert self.filter_opt_corr is not None
		if predata == "vwap":
			self.period_vwap = int(default_radar['default_bf_period_vwap']) if self.period_vwap is None else self.period_vwap
			self.percentage_range = float(default_radar['default_radar_percentage_range']) if self.percentage_range is None else self.percentage_range
			self.period_predata = self.radar_period + self.period_vwap
		else:
			self.period_predata = 0

		# Get  filtered_stock that should be analyzed
		self.filtered_stockcodes = await self._get_stockcodes(
			screener_min_value=self.screener_min_value,
			screener_min_frequency=self.screener_min_frequency,
			stockcode_excludes=self.stockcode_excludes,
			dbs=self.dbs)
		
		# Get raw data
		raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumvol, raw_data_broker_sumval, self.filtered_stockcodes = \
			await self._get_stock_raw_data(
				filtered_stockcodes=self.filtered_stockcodes,
				startdate=self.startdate,
				enddate=self.enddate,
				default_months_range=self.default_months_range,
				dbs=self.dbs
				)
		
		# Get broker flow parameters for each stock in filtered_stockcodes
		self.selected_broker, self.optimum_n_selected_cluster, self.optimum_corr, self.broker_features = \
			await self._get_bf_parameters(
				raw_data_close=raw_data_full["close"],
				raw_data_broker_nval=raw_data_broker_nval,
				raw_data_broker_sumval=raw_data_broker_sumval,
				n_selected_cluster=self.n_selected_cluster,
				training_start_index=self.training_start_index,
				training_end_index=self.training_end_index,
				splitted_min_n_cluster=self.splitted_min_n_cluster,
				splitted_max_n_cluster=self.splitted_max_n_cluster,
			)
		
		# Filter code based on self.optimum_corr should be greater than self.filter_opt_corr
		self.filtered_stockcodes, raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval = \
			await self._get_filtered_stockcodes_by_corr(
				filter_opt_corr=self.filter_opt_corr,
				optimum_corr=self.optimum_corr,
				filtered_stockcodes=self.filtered_stockcodes,
				raw_data_full=raw_data_full,
				raw_data_broker_nvol=raw_data_broker_nvol,
				raw_data_broker_nval=raw_data_broker_nval,
				raw_data_broker_sumval=raw_data_broker_sumval,
			)
		
		# Get radar period filtered stockdata
		self.startdate, self.enddate, self.raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval = \
			await self._get_radar_period_filtered_stock_data(
				startdate = self.startdate,
				radar_period = self.radar_period,
				period_predata = self.period_predata,
				raw_data_full = raw_data_full,
				raw_data_broker_nvol = raw_data_broker_nvol,
				raw_data_broker_nval = raw_data_broker_nval,
				raw_data_broker_sumval = raw_data_broker_sumval,
			)
		
		# Get sum of selected broker transaction for each stock
		self.selected_broker_nvol, self.selected_broker_nval, self.selected_broker_sumval = \
			await self._sum_selected_broker_transaction(
				raw_data_broker_nvol = raw_data_broker_nvol,
				raw_data_broker_nval = raw_data_broker_nval,
				raw_data_broker_sumval = raw_data_broker_sumval,
				selected_broker = self.selected_broker
			)

		return self
	
class ScreenerMoneyFlow(ScreenerBase):
	def __init__(self,
		accum_or_distri: Literal[dp.ScreenerList.most_accumulated,dp.ScreenerList.most_distributed] = dp.ScreenerList.most_accumulated,
		n_stockcodes: int = 10,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		stockcode_excludes: set[str] = set(),
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		n_selected_cluster:int | None = None,
		radar_period: int | None = None,
		period_mf: int | None = None,
		period_pricecorrel: int | None = None,
		default_months_range: int | None = None,
		training_start_index: float | None = None,
		training_end_index: float | None = None,
		min_n_cluster: int | None = None,
		max_n_cluster: int | None = None,
		splitted_min_n_cluster: int | None = None,
		splitted_max_n_cluster: int | None = None,
		stepup_n_cluster_threshold: int | None = None,
		filter_opt_corr: int | None = None,
		dbs: db.Session = next(db.get_dbs()),
		) -> None:
		
		super().__init__(
			startdate=startdate,
			enddate=enddate,
			stockcode_excludes=stockcode_excludes,
			screener_min_value=screener_min_value,
			screener_min_frequency=screener_min_frequency,
			n_selected_cluster=n_selected_cluster,
			radar_period=radar_period,
			period_mf=period_mf,
			period_pricecorrel=period_pricecorrel,
			default_months_range=default_months_range,
			training_start_index=training_start_index,
			training_end_index=training_end_index,
			min_n_cluster=min_n_cluster,
			max_n_cluster=max_n_cluster,
			splitted_min_n_cluster=splitted_min_n_cluster,
			splitted_max_n_cluster=splitted_max_n_cluster,
			stepup_n_cluster_threshold=stepup_n_cluster_threshold,
			filter_opt_corr=filter_opt_corr,
			dbs=dbs,
		)

		self.accum_or_distri = accum_or_distri
		self.n_stockcodes = n_stockcodes

	async def screen(self) -> ScreenerMoneyFlow:
		# get default param radar, defined startdate,
		# filtered_stockcodes should be analyzed, and the selected brokers each stock
		await super()._fit_base()

		# get ranked, filtered, and pre-calculated indicator of filtered_stockcodes
		self.top_stockcodes = await self._get_mf_top_stockcodes(
			accum_or_distri=self.accum_or_distri,
			n_stockcodes=self.n_stockcodes,
			startdate=self.startdate,
			enddate=self.enddate,
		)
		return self
	
	async def _get_mf_top_stockcodes(self,
		accum_or_distri: dp.ScreenerList = dp.ScreenerList.most_accumulated,
		n_stockcodes: int = 10,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today()
		) -> pd.DataFrame:
		top_stockcodes = pd.DataFrame()
		
		# Sum of selected broker transaction for each stock
		# Get only self.selected_broker_nval between startdate and enddate based on level 1 date index
		if accum_or_distri == dp.ScreenerList.most_distributed:
			top_stockcodes['mf'] = self.selected_broker_nval.loc[
				self.selected_broker_nval.index.get_level_values(1).isin(pd.date_range(start=startdate, end=enddate))
				].groupby("code").sum().nsmallest(n=n_stockcodes, columns="broker_nval")['broker_nval']
		else:
			top_stockcodes['mf'] = self.selected_broker_nval.loc[
				self.selected_broker_nval.index.get_level_values(1).isin(pd.date_range(start=startdate, end=enddate))
				].groupby("code").sum().nlargest(n=n_stockcodes, columns="broker_nval")['broker_nval']

		# get selected_broker_nval that has level 0 index (code) in top_stockcodes
		self.selected_broker_nval = self.selected_broker_nval[self.selected_broker_nval.index.get_level_values(0).isin(top_stockcodes.index)]
		self.selected_broker_sumval = self.selected_broker_sumval[self.selected_broker_sumval.index.get_level_values(0).isin(top_stockcodes.index)]
		self.raw_data_full = self.raw_data_full[self.raw_data_full.index.get_level_values(0).isin(top_stockcodes.index)]
		self.bar_range = int(self.raw_data_full.groupby(level='code').size().max())
		
		# Calculate top_stockcodes Prop. Get only between startdate and enddate based on level 1 date index
		top_stockcodes['prop'] = (self.selected_broker_sumval.loc[
				self.selected_broker_sumval.index.get_level_values(1).isin(pd.date_range(start=startdate, end=enddate))
			]['broker_sumval'].groupby("code").sum())\
			/(self.raw_data_full.loc[
				self.raw_data_full.index.get_level_values(1).isin(pd.date_range(start=startdate, end=enddate))
			]['value'].groupby("code").sum()*2)
		
		# Calculate top_stockcodes PriceCorrel
		if (startdate == enddate) or (self.radar_period == 1):
			wvalflow = self.selected_broker_nval['broker_nval'].groupby("code").cumsum()
			top_stockcodes['pricecorrel'] = wvalflow.groupby("code").corr(self.raw_data_full['close']) # type: ignore
		else:
			wvalflow = self.selected_broker_nval.loc[
				self.selected_broker_nval.index.get_level_values(1).isin(pd.date_range(start=startdate, end=enddate))
				]['broker_nval'].groupby("code").cumsum()
			
			top_stockcodes['pricecorrel'] = wvalflow.groupby("code").corr( # type: ignore
				self.raw_data_full.loc[
					self.raw_data_full.index.get_level_values(1).isin(pd.date_range(start=startdate, end=enddate))
				]['close'])

		return top_stockcodes

class ScreenerVWAP(ScreenerBase):
	"""
	Get the top n stockcodes based on VWAP indicator:
		- Rally (always close > vwap within n days)
		- Around VWAP (close around x% of vwap)
		- Breakout (t_x: close < vwap, t_y: close > vwap, within n days, and now close > vwap)
		- Breakdown (t_x: close > vwap, t_y: close < vwap, within n days, and now close < vwap)
	
	General Rules:
		- Value, Freq, Prop more than min_value, min_frequency, min_prop
	"""
	def __init__(self,
		screener_vwap_criteria: Literal[dp.ScreenerList.vwap_rally,dp.ScreenerList.vwap_around,dp.ScreenerList.vwap_breakout,dp.ScreenerList.vwap_breakdown] = dp.ScreenerList.vwap_rally,
		n_stockcodes: int = 10,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		radar_period: int | None = None,
		percentage_range: float | None = 0.05,
		period_vwap: int | None = None,
		stockcode_excludes: set[str] = set(),
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		n_selected_cluster:int | None = None,
		period_mf: int | None = None,
		period_pricecorrel: int | None = None,
		default_months_range: int | None = None,
		training_start_index: float | None = None,
		training_end_index: float | None = None,
		min_n_cluster: int | None = None,
		max_n_cluster: int | None = None,
		splitted_min_n_cluster: int | None = None,
		splitted_max_n_cluster: int | None = None,
		stepup_n_cluster_threshold: int | None = None,
		filter_opt_corr: int | None = None,
		dbs: db.Session = next(db.get_dbs())
		) -> None:
		screener_vwap_criteria_list = [
			dp.ScreenerList.vwap_rally,
			dp.ScreenerList.vwap_around,
			dp.ScreenerList.vwap_breakout,
			dp.ScreenerList.vwap_breakdown
		]
		assert screener_vwap_criteria in screener_vwap_criteria_list, f'screener_vwap_criteria_list must be {screener_vwap_criteria_list}'

		super().__init__(
			startdate=startdate,
			enddate=enddate,
			stockcode_excludes=stockcode_excludes,
			screener_min_value=screener_min_value,
			screener_min_frequency=screener_min_frequency,
			n_selected_cluster=n_selected_cluster,
			radar_period=radar_period,
			period_mf=period_mf,
			period_pricecorrel=period_pricecorrel,
			default_months_range=default_months_range,
			training_start_index=training_start_index,
			training_end_index=training_end_index,
			min_n_cluster=min_n_cluster,
			max_n_cluster=max_n_cluster,
			splitted_min_n_cluster=splitted_min_n_cluster,
			splitted_max_n_cluster=splitted_max_n_cluster,
			stepup_n_cluster_threshold=stepup_n_cluster_threshold,
			filter_opt_corr=filter_opt_corr,
			dbs=dbs,
		)

		self.screener_vwap_criteria = screener_vwap_criteria
		self.n_stockcodes = n_stockcodes
		self.radar_period = radar_period
		self.percentage_range = percentage_range
		self.period_vwap = period_vwap
	
	async def screen(self) -> ScreenerVWAP:
		# get default param radar, defined startdate,
		# filtered_stockcodes should be analyzed
		# selected_broker each stock, also optimum_n_selected_cluster, optimum_corr, broker_features
		# raw_data_full, selected_broker_nvol, selected_broker_nval, selected_broker_sumval
		await super()._fit_base(predata="vwap")
		assert isinstance(self.period_vwap, int), f'period_vwap must be integer'
		assert isinstance(self.percentage_range, float), f'percentage_range must be float'

		# combine raw_data_full, selected_broker_nval, selected_broker_nvol, selected_broker_sumval
		self.raw_data_full = self.raw_data_full.join([self.selected_broker_nval, self.selected_broker_nvol, self.selected_broker_sumval], how='left')
		
		# Get self.raw_data_full['vwap'] inside date range
		self.raw_data_full, self.bar_range = await self._vwap_prepare(raw_data_full=self.raw_data_full, period_vwap=self.period_vwap, startdate=self.startdate, enddate=self.enddate)

		# Go to get top codes for each screener_vwap_criteria
		if self.screener_vwap_criteria == dp.ScreenerList.vwap_rally:
			stocklist = await self._get_vwap_rally(raw_data_full=self.raw_data_full)
		elif self.screener_vwap_criteria == dp.ScreenerList.vwap_around:
			stocklist = await self._get_vwap_around(raw_data_full=self.raw_data_full, percentage_range=self.percentage_range)
		elif self.screener_vwap_criteria == dp.ScreenerList.vwap_breakout:
			stocklist = await self._get_vwap_breakout(raw_data_full=self.raw_data_full)
		elif self.screener_vwap_criteria == dp.ScreenerList.vwap_breakdown:
			stocklist = await self._get_vwap_breakdown(raw_data_full=self.raw_data_full)
		else:
			raise ValueError(f'Invalid screener_vwap_criteria: {self.screener_vwap_criteria}')
		
		# Get data from stocklist
		self.stocklist, self.top_data = await self._get_data_from_stocklist(stocklist)

		# Compile data for top_stockcodes from stocklist and top_data
		self.top_stockcodes = self.top_data[['close','vwap']].groupby(level='code').last()
		self.top_stockcodes['mf'] = self.top_data['broker_nval'].groupby(level='code').sum()
		self.top_stockcodes = self.top_stockcodes.sort_values('mf', ascending=False)
		
		return self

	async def _vwap_prepare(self,
		raw_data_full: pd.DataFrame = ...,
		period_vwap: int = ...,
		startdate: datetime.date = ...,
		enddate: datetime.date = ...,
		)-> tuple[pd.DataFrame, int]:
		# Whale-VWAP
		raw_data_full['vwap'] = ((raw_data_full['broker_nval'].groupby(level='code').rolling(window=period_vwap).apply(lambda x: x[x>0].sum()))\
			/(raw_data_full['broker_nvol'].groupby(level='code').rolling(window=period_vwap).apply(lambda x: x[x>0].sum()))).droplevel(0)
		raw_data_full['vwap'] = raw_data_full['vwap'].mask(raw_data_full['vwap'].le(0)).ffill()

		# Filter only startdate to enddate
		raw_data_full = raw_data_full.loc[
			(raw_data_full.index.get_level_values('date') >= pd.Timestamp(startdate)) & \
			(raw_data_full.index.get_level_values('date') <= pd.Timestamp(enddate))]

		bar_range = int(raw_data_full.groupby(level='code').size().max())

		return raw_data_full, bar_range
	
	async def _get_data_from_stocklist(self, stocklist: list) -> tuple[list, pd.DataFrame]:
		# Get data from stocklist
		top_data = self.raw_data_full.loc[self.raw_data_full.index.get_level_values('code').isin(stocklist)]
		# Sum broker_nval for each code and get top n_stockcodes
		stocklist = top_data['broker_nval'].groupby(level='code').sum().nlargest(self.n_stockcodes).index.tolist()

		# Get data from stocklist
		top_data = self.raw_data_full.loc[self.raw_data_full.index.get_level_values('code').isin(stocklist)]
		
		return stocklist, top_data

	async def _get_vwap_rally(self, raw_data_full: pd.DataFrame) -> list:
		"""Rally (always close > vwap within n days)"""
		# Get stockcodes with raw_data_full['close'] always raw_data_full['vwap']
		stocklist = (raw_data_full['close'] >= raw_data_full['vwap']).groupby(level='code').all()
		stocklist = stocklist[stocklist].index.tolist()

		return stocklist

	async def _get_vwap_around(self, raw_data_full: pd.DataFrame, percentage_range: float) -> list:
		"""Around VWAP (close around x% of vwap)"""
		# Get stockcodes with last raw_data_full['close'] around last raw_data_full['vwap'], within percentage_range
		last_data = raw_data_full[['close','vwap']].groupby(level='code').last()
		stocklist = last_data[(last_data['close'] >= last_data['vwap']*(1-percentage_range)) & (last_data['close'] <= last_data['vwap']*(1+percentage_range))].index.tolist()

		return stocklist

	async def _get_vwap_breakout(self, raw_data_full: pd.DataFrame) -> list:
		"""Breakout (t_(x-1): close < vwap, t_(x): close > vwap, within n days, and now close > vwap)"""
		# Get stockcodes with now close > vwap
		last_data = raw_data_full[['close','vwap']].groupby(level='code').last()
		stocklist = last_data[last_data['close'] >= last_data['vwap']].index.tolist()

		# Define breakout
		top_data = raw_data_full.loc[raw_data_full.index.get_level_values('code').isin(stocklist)]
		top_data['close_morethan_vwap'] = top_data['close'] >= top_data['vwap']
		top_data['breakout'] = top_data.groupby(level='code').rolling(window=2)['close_morethan_vwap']\
			.apply(lambda x: (x.iloc[0] == False) & (x.iloc[1] == True)).droplevel(0)
		
		# Get stockcodes with breakout
		stocklist = top_data['breakout'].groupby(level='code').any()
		stocklist = stocklist[stocklist].index.tolist()

		return stocklist

	async def _get_vwap_breakdown(self, raw_data_full: pd.DataFrame) -> list:
		"""Breakdown (t_x: close > vwap, t_y: close < vwap, within n days, and now close < vwap)"""
		# Get stockcodes with now close < vwap
		last_data = raw_data_full[['close','vwap']].groupby(level='code').last()
		stocklist = last_data[last_data['close'] <= last_data['vwap']].index.tolist()

		# Define breakdown
		top_data = raw_data_full.loc[raw_data_full.index.get_level_values('code').isin(stocklist)]
		top_data['close_lessthan_vwap'] = top_data['close'] <= top_data['vwap']
		top_data['breakdown'] = top_data.groupby(level='code').rolling(window=2)['close_lessthan_vwap']\
			.apply(lambda x: (x.iloc[0] == False) & (x.iloc[1] == True)).droplevel(0)
		
		# Get stockcodes with breakdown
		stocklist = top_data['breakdown'].groupby(level='code').any()
		stocklist = stocklist[stocklist].index.tolist()

		return stocklist