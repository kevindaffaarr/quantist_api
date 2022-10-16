"""
Broker Flow Module by Quantist.io
"""

from __future__ import annotations
import gc

import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
		enddate: datetime.date | None = datetime.date.today(),
		n_selected_cluster:int | None = None,
		period_wmf: int | None = None,
		period_wprop: int | None = None,
		period_wpricecorrel: int | None = None,
		period_wmapricecorrel: int | None = None,
		period_wvwap:int | None = None,
		wpow_high_wprop: int | None = None,
		wpow_high_wpricecorrel: int | None = None,
		wpow_high_wmapricecorrel: int | None = None,
		wpow_medium_wprop: int | None = None,
		wpow_medium_wpricecorrel: int | None = None,
		wpow_medium_wmapricecorrel: int | None = None,
		training_start_index: int | None = None,
		training_end_index: int | None = None,
		min_n_cluster: int | None = None,
		max_n_cluster: int | None = None,
		splitted_min_n_cluster: int | None = None,
		splitted_max_n_cluster: int | None = None,
		stepup_n_cluster_threshold: int | None = None,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> None:
		self.stockcode = stockcode
		self.startdate = startdate
		self.enddate = enddate
		self.n_selected_cluster = n_selected_cluster
		self.period_wmf = period_wmf
		self.period_wprop = period_wprop
		self.period_wpricecorrel = period_wpricecorrel
		self.period_wmapricecorrel = period_wmapricecorrel
		self.period_wvwap = period_wvwap
		self.wpow_high_wprop = wpow_high_wprop
		self.wpow_high_wpricecorrel = wpow_high_wpricecorrel
		self.wpow_high_wmapricecorrel = wpow_high_wmapricecorrel
		self.wpow_medium_wprop = wpow_medium_wprop
		self.wpow_medium_wpricecorrel = wpow_medium_wpricecorrel
		self.wpow_medium_wmapricecorrel = wpow_medium_wmapricecorrel
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
		self.bf_indicators = None

	async def fit(self) -> StockBFFull:
		# Get default bf params
		default_bf = await self.__get_default_bf(dbs=self.dbs)

		# Check Does Stock Code is composite
		# Default stock is parameterized, may become a branding or endorsement option
		self.stockcode = (str(default_bf['default_stockcode']) if self.stockcode is None else self.stockcode).lower()
		if self.stockcode == 'composite':
			raise ValueError("Broker Flow is not available yet for index")
		else:
			qry = self.dbs.query(db.ListStock.code).filter(db.ListStock.code == self.stockcode)
			row = pd.read_sql(sql=qry.statement, con=self.dbs.bind)
			if len(row) == 0:
				raise KeyError("There is no such stock code in the database.")
		# Data Parameter
		default_months_range = int(default_bf['default_months_range']) if self.startdate is None else 0
		self.enddate = datetime.date.today() if self.enddate is None else self.enddate
		self.startdate = self.enddate - relativedelta(months=default_months_range) if self.startdate is None else self.startdate
		self.period_wmf = int(default_bf['default_bf_period_wmf']) if self.period_wmf is None else self.period_wmf
		self.period_wprop = int(default_bf['default_bf_period_wprop']) if self.period_wprop is None else self.period_wprop
		self.period_wpricecorrel = int(default_bf['default_bf_period_wpricecorrel']) if self.period_wpricecorrel is None else self.period_wpricecorrel
		self.period_wmapricecorrel = int(default_bf['default_bf_period_wmapricecorrel']) if self.period_wmapricecorrel is None else self.period_wmapricecorrel
		self.period_wvwap = int(default_bf['default_bf_period_wvwap']) if self.period_wvwap is None else self.period_wvwap
		self.wpow_high_wprop = int(default_bf['default_bf_wpow_high_wprop']) if self.wpow_high_wprop is None else self.wpow_high_wprop
		self.wpow_high_wpricecorrel = int(default_bf['default_bf_wpow_high_wpricecorrel']) if self.wpow_high_wpricecorrel is None else self.wpow_high_wpricecorrel
		self.wpow_high_wmapricecorrel = int(default_bf['default_bf_wpow_high_wmapricecorrel']) if self.wpow_high_wmapricecorrel is None else self.wpow_high_wmapricecorrel
		self.wpow_medium_wprop = int(default_bf['default_bf_wpow_medium_wprop']) if self.wpow_medium_wprop is None else self.wpow_medium_wprop
		self.wpow_medium_wpricecorrel = int(default_bf['default_bf_wpow_medium_wpricecorrel']) if self.wpow_medium_wpricecorrel is None else self.wpow_medium_wpricecorrel
		self.wpow_medium_wmapricecorrel = int(default_bf['default_bf_wpow_medium_wmapricecorrel']) if self.wpow_medium_wmapricecorrel is None else self.wpow_medium_wmapricecorrel
		preoffset_period_param = max(self.period_wmf,self.period_wprop,self.period_wpricecorrel,(self.period_wmapricecorrel+self.period_wvwap))-1

		self.training_start_index = int(default_bf['default_bf_training_start_index'])/100 if self.training_start_index is None else self.training_start_index/100
		self.training_end_index = int(default_bf['default_bf_training_end_index'])/100 if self.training_end_index is None else self.training_end_index/100
		self.min_n_cluster = int(default_bf['default_bf_min_n_cluster']) if self.min_n_cluster is None else self.min_n_cluster
		self.max_n_cluster = int(default_bf['default_bf_max_n_cluster']) if self.max_n_cluster is None else self.max_n_cluster
		self.splitted_min_n_cluster = int(default_bf['default_bf_splitted_min_n_cluster']) if self.splitted_min_n_cluster is None else self.splitted_min_n_cluster
		self.splitted_max_n_cluster = int(default_bf['default_bf_splitted_max_n_cluster']) if self.splitted_max_n_cluster is None else self.splitted_max_n_cluster
		self.stepup_n_cluster_threshold = int(default_bf['default_bf_stepup_n_cluster_threshold'])/100 if self.stepup_n_cluster_threshold is None else self.stepup_n_cluster_threshold/100

		# Get full stockdatatransaction
		raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval = \
			await self.__get_stock_raw_data(
				stockcode=self.stockcode,
				startdate=self.startdate,
				enddate=self.enddate,
				preoffset_period_param=preoffset_period_param,
				dbs=self.dbs
				)
		# Get broker flow parameters
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
			)

		# Calc broker flow indicators
		self.bf_indicators = await self.calc_bf_indicators(
			raw_data_full = raw_data_full,
			raw_data_broker_nvol = raw_data_broker_nvol,
			raw_data_broker_nval = raw_data_broker_nval,
			raw_data_broker_sumval = raw_data_broker_sumval,
			selected_broker = self.selected_broker,
			period_wmf = self.period_wmf,
			period_wprop = self.period_wprop,
			period_wpricecorrel = self.period_wpricecorrel,
			period_wmapricecorrel = self.period_wmapricecorrel,
			period_wvwap = self.period_wvwap,
			wpow_high_wprop = self.wpow_high_wprop,
			wpow_high_wpricecorrel = self.wpow_high_wpricecorrel,
			wpow_high_wmapricecorrel = self.wpow_high_wmapricecorrel,
			wpow_medium_wprop = self.wpow_medium_wprop,
			wpow_medium_wpricecorrel = self.wpow_medium_wpricecorrel,
			wpow_medium_wmapricecorrel = self.wpow_medium_wmapricecorrel,
			preoffset_period_param = preoffset_period_param,
		)

		return self

	async def __get_default_bf(self,dbs: db.Session | None = next(db.get_dbs())) -> pd.Series:
		# Get Default Broker Flow
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter((db.DataParam.param.like("default_bf_%")) | \
				(db.DataParam.param.like("default_stockcode")) | \
				(db.DataParam.param.like("default_months_range")))
		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])

	# Get Net Val Sum Val Broker Transaction
	async def __get_nvsv_broker_transaction(self,raw_data_broker_full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
		# Calculate nval (bval-sval) and sumval (bval+sval)
		raw_data_broker_full["nvol"] = raw_data_broker_full["bvol"] - raw_data_broker_full["svol"]
		raw_data_broker_full["nval"] = raw_data_broker_full["bval"] - raw_data_broker_full["sval"]
		raw_data_broker_full["sumval"] = raw_data_broker_full["bval"] + raw_data_broker_full["sval"]
		# Aggretate by broker then broker to column for each net and sum
		raw_data_broker_nvol = raw_data_broker_full.pivot(index=None,columns="broker",values="nvol")
		raw_data_broker_nval = raw_data_broker_full.pivot(index=None,columns="broker",values="nval")
		raw_data_broker_sumval = raw_data_broker_full.pivot(index=None,columns="broker",values="sumval")

		# Fill na
		raw_data_broker_nvol.fillna(value=0, inplace=True)
		raw_data_broker_nval.fillna(value=0, inplace=True)
		raw_data_broker_sumval.fillna(value=0, inplace=True)

		# Return
		return raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval

	async def __get_full_broker_transaction(self,
		stockcode: str,
		preoffset_startdate: datetime.date | None = None,
		enddate: datetime.date | None = datetime.date.today(),
		default_months_range: int | None = 12,
		dbs: db.Session | None = next(db.get_dbs()),
		):
		# if startdate is none, set to 1 year before enddate
		if preoffset_startdate is None:
			preoffset_startdate = enddate - relativedelta(months=default_months_range)

		# Query Definition
		qry = dbs.query(
			db.StockTransaction.date,
			db.StockTransaction.broker,
			db.StockTransaction.bvol,
			db.StockTransaction.svol,
			db.StockTransaction.bval,
			db.StockTransaction.sval
		).filter((db.StockTransaction.code == stockcode))

		# Main Query
		qry_main = qry.filter(db.StockTransaction.date.between(preoffset_startdate, enddate))

		# Main Query Fetching
		raw_data_broker_full = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by=["date","broker"], ascending=[True,True])\
			.reset_index(drop=True).set_index("date")

		# Data Cleansing: fillna
		raw_data_broker_full.fillna(value=0, inplace=True)

		return raw_data_broker_full

	async def __get_stock_price_data(self,
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		preoffset_period_param: int | None = 50,
		default_months_range: int | None = 12,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> pd.Series:
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
		qry_main = qry.filter(db.StockData.date.between(startdate, enddate))

		# Main Query Fetching
		raw_data_main = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True).set_index('date')
		# Update self.startdate to available date in database

		# Pre-Data Query
		startdate = raw_data_main.index[0].date()
		qry_pre = qry.filter(db.StockData.date < startdate)\
			.order_by(db.StockData.date.desc())\
			.limit(preoffset_period_param)

		# Pre-Data Query Fetching
		raw_data_pre = pd.read_sql(sql=qry_pre.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True).set_index('date')

		# Concatenate Pre and Main Query
		raw_data_full = pd.concat([raw_data_pre,raw_data_main])
		
		if len(raw_data_pre) > 0:
			self.preoffset_startdate = raw_data_pre.index[0].date()
		else:
			self.preoffset_startdate = startdate

		# Data Cleansing: zero openprice replace with previous
		raw_data_full['openprice'] = raw_data_full['openprice'].mask(raw_data_full['openprice'].eq(0),raw_data_full['previous'])

		# End of Method: Return or Assign Attribute
		return raw_data_full

	async def __get_stock_raw_data(self,
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_months_range: int | None = 12,
		preoffset_period_param: int | None = 50,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		# Get Stockdata Full
		raw_data_full = await self.__get_stock_price_data(
			stockcode=stockcode,startdate=startdate,enddate=enddate,
			preoffset_period_param=preoffset_period_param,
			default_months_range=default_months_range,dbs=dbs)

		# Get Raw Data Broker Full
		raw_data_broker_full = await self.__get_full_broker_transaction(
			stockcode=stockcode,preoffset_startdate=self.preoffset_startdate,enddate=enddate,
			default_months_range=default_months_range,dbs=dbs)

		# Transform Raw Data Broker
		raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval = await self.__get_nvsv_broker_transaction(raw_data_broker_full=raw_data_broker_full)

		return raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval

	#TODO get composite raw data def: __get_composite_raw_data ()

	async def __get_selected_broker(self,
		clustered_features: pd.DataFrame,
		centroids_cluster: pd.DataFrame,
		n_selected_cluster: int | None = 1,
		) -> list[str]:
		# Get index of max value in column 0 in centroid
		selected_cluster = (centroids_cluster[0]).nlargest(n_selected_cluster).index.tolist()
		# selected_cluster = (abs(centroids_cluster[0]*centroids_cluster[1])).nlargest(n_selected_cluster).index.tolist()
		
		# Get sorted selected broker
		selected_broker = clustered_features.loc[clustered_features["cluster"].isin(selected_cluster), :]\
			.sort_values(by="corr_ncum_close", ascending=False)\
			.index.tolist()

		return selected_broker

	async def __get_selected_broker_ncum(self,
		selected_broker: list[str],
		broker_ncum: pd.DataFrame,
		) -> pd.Series:
		# Get selected broker transaction by columns of net_stockdatatransaction, then sum each column to aggregate to date
		selected_broker_ncum = broker_ncum[selected_broker].sum(axis=1)
		# Get cumulative
		selected_broker_ncum = selected_broker_ncum.cumsum().rename("selected_broker_ncum")
		return selected_broker_ncum

	async def __get_corr_selected_broker_ncum(self,
		clustered_features: pd.DataFrame,
		raw_data_close: pd.Series,
		broker_ncum: pd.Series,
		centroids_cluster: pd.DataFrame,
		n_selected_cluster: int | None = None,
		) -> pd.Series:
		selected_broker = await self.__get_selected_broker(
			clustered_features=clustered_features,
			centroids_cluster=centroids_cluster,
			n_selected_cluster=n_selected_cluster
			)

		# Get selected broker transaction by columns of net_stockdatatransaction, then sum each column to aggregate to date
		selected_broker_ncum = await self.__get_selected_broker_ncum(selected_broker, broker_ncum)

		# Return correlation between close and selected_broker_ncum
		return selected_broker_ncum.corr(raw_data_close)

	async def __optimize_selected_cluster(self,
		clustered_features: pd.DataFrame,
		raw_data_close: pd.Series,
		broker_ncum: pd.Series,
		centroids_cluster: pd.DataFrame,
		stepup_n_cluster_threshold: float | None = 0.05,
		n_selected_cluster: int | None = None,
		) -> tuple[list[str], int, float]:
		# Check does n_selected_cluster already defined
		if n_selected_cluster is None:
			# Define correlation param
			corr_list = []

			# Iterate optimum n_cluster
			for n_selected_cluster in range(1,len(centroids_cluster)+1):
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
			index_max_corr: int = np.argmax(corr_list)
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
		min_n_cluster:int | None = 4,
		max_n_cluster:int | None = 10,
		) -> tuple[pd.DataFrame, pd.DataFrame]:

		# Get X and Y
		X = features[[x,y]].values
		# Define silhouette param
		silhouette_coefficient = []
		max_n_cluster = min(max_n_cluster, len(X)-1)

		# Iterate optimum n_cluster
		for n_cluster in range(min_n_cluster, max_n_cluster+1):
			# Clustering
			kmeans = KMeans(init="k-means++", n_clusters=n_cluster, random_state=0).fit(X)
			score = silhouette_score(X, kmeans.labels_)
			silhouette_coefficient.append(score)
		# Define optimum n_cluster
		optimum_n_cluster = np.argmax(silhouette_coefficient) + min_n_cluster

		# Clustering with optimum n cluster
		kmeans = KMeans(init="k-means++", n_clusters=optimum_n_cluster, random_state=0).fit(X)
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
		training_start_index: float | None = 0.5,
		training_end_index: float | None = 0.75,
		splitted_min_n_cluster: int | None = 2,
		splitted_max_n_cluster: int | None = 5,
		stepup_n_cluster_threshold: float | None = 0.05,
		) -> tuple[list[str], int, float, pd.DataFrame]:

		# Only get third quartile of raw_data so not over-fitting
		length = len(raw_data_close)
		start_index = int(length*training_start_index)
		end_index = int(length*training_end_index)
		raw_data_close = raw_data_close.iloc[start_index:end_index]
		raw_data_broker_nval = raw_data_broker_nval.iloc[start_index:end_index,:]
		raw_data_broker_sumval = raw_data_broker_sumval.iloc[start_index:end_index,:]

		if (raw_data_broker_nval == 0).all().all() or (raw_data_broker_nval == 0).all().all():
			raise ValueError("There is no transaction for the stockcode in the selected quantile")
		
		# Cumulate value for nval
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

	async def calc_bf_indicators(self,
		raw_data_full: pd.DataFrame,
		raw_data_broker_nvol: pd.DataFrame,
		raw_data_broker_nval: pd.DataFrame,
		raw_data_broker_sumval: pd.DataFrame,
		selected_broker: list[str],
		period_wmf: int | None = 1,
		period_wprop: int | None = 10,
		period_wpricecorrel: int | None = 10,
		period_wmapricecorrel: int | None = 100,
		period_wvwap:int | None = 21,
		wpow_high_wprop: int | None = 40,
		wpow_high_wpricecorrel: int | None = 50,
		wpow_high_wmapricecorrel: int | None = 30,
		wpow_medium_wprop: int | None = 20,
		wpow_medium_wpricecorrel: int | None = 30,
		wpow_medium_wmapricecorrel: int | None = 30,
		preoffset_period_param: int | None = 50,
		) -> pd.DataFrame:
		# OHLC
		# raw_data_full

		# Broker Data Prep
		selected_data_broker_nvol = raw_data_broker_nvol[selected_broker].sum(axis=1)
		selected_data_broker_nval = raw_data_broker_nval[selected_broker].sum(axis=1)
		selected_data_broker_sumval = raw_data_broker_sumval[selected_broker].sum(axis=1)

		# Net Value for Volume Profile
		raw_data_full["netvol"] = selected_data_broker_nvol
		
		# Whale Volume Flow
		raw_data_full["wvolflow"] = selected_data_broker_nvol.cumsum()

		# Whale Money Flow
		raw_data_full['wmf'] = selected_data_broker_nval.rolling(window=period_wmf).sum()

		# Whale Proportion
		raw_data_full['wprop'] = selected_data_broker_sumval.rolling(window=period_wprop).sum() \
			/ (raw_data_full['value'].rolling(window=period_wprop).sum()*2)

		# Whale Net Proportion
		raw_data_full['wnetprop'] = selected_data_broker_nval.rolling(window=period_wprop).sum() \
			/ (raw_data_full['value'].rolling(window=period_wprop).sum()*2)

		# Whale correlation
		raw_data_full['wpricecorrel'] = raw_data_full["wvolflow"].rolling(window=period_wpricecorrel).corr(raw_data_full['close'])

		# Whale MA correlation
		raw_data_full['wmapricecorrel'] = raw_data_full['wpricecorrel'].rolling(window=period_wmapricecorrel).mean()

		# Whale-VWAP
		raw_data_full['wvwap'] = (selected_data_broker_nval.rolling(window=period_wvwap).apply(lambda x: x[x>0].sum()))\
			/(selected_data_broker_nvol.rolling(window=period_wvwap).apply(lambda x: x[x>0].sum()))
		raw_data_full['wvwap'] = raw_data_full['wvwap'].mask(raw_data_full['wvwap'].le(0)).ffill()

		# Whale Power
		raw_data_full['wpow'] = \
			np.where(
				(raw_data_full["wprop"]>(wpow_high_wprop/100)) & \
				(raw_data_full['wpricecorrel']>(wpow_high_wpricecorrel/100)) & \
				(raw_data_full['wmapricecorrel']>(wpow_high_wmapricecorrel/100)), \
				3,
				np.where(
					(raw_data_full["wprop"]>(wpow_medium_wprop/100)) & \
					(raw_data_full['wpricecorrel']>(wpow_medium_wpricecorrel/100)) & \
					(raw_data_full['wmapricecorrel']>(wpow_medium_wmapricecorrel/100)), \
					2, \
					1
				)
			)

		# End of Method: Return Processed Raw Data to BF Indicators
		return raw_data_full.drop(raw_data_full.index[:preoffset_period_param])

	async def chart(self,media_type: str | None = None):
		fig = await genchart.broker_chart(
			self.stockcode,self.bf_indicators,
			self.selected_broker,
			self.optimum_n_selected_cluster,
			self.optimum_corr,
			self.period_wprop,
			self.period_wpricecorrel,
			self.period_wmapricecorrel,
			self.period_wvwap,
			)
		if media_type in ["png","jpeg","jpg","webp","svg"]:
			return await genchart.fig_to_image(fig,media_type)
		elif media_type == "json":
			return await genchart.fig_to_json(fig)
		else:
			return fig
	
	async def broker_cluster_chart(self,media_type: str | None = None):
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
		enddate: datetime.date | None = datetime.date.today(),
		y_axis_type: dp.ListRadarType | None = dp.ListRadarType.correlation,
		stockcode_excludes: set[str] | None = set(),
		include_composite: bool | None = False,
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		n_selected_cluster:int | None = None,
		radar_period: int | None = None,
		period_wmf: int | None = None,
		period_wpricecorrel: int | None = None,
		default_months_range: int | None = None,
		training_start_index: int | None = None,
		training_end_index: int | None = None,
		min_n_cluster: int | None = None,
		max_n_cluster: int | None = None,
		splitted_min_n_cluster: int | None = None,
		splitted_max_n_cluster: int | None = None,
		stepup_n_cluster_threshold: int | None = None,
		filter_opt_corr: int | None = None,
		dbs: db.Session | None = next(db.get_dbs()),
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
		self.period_wmf = period_wmf
		self.period_wpricecorrel = period_wpricecorrel
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

		self.radar_indicators = None
	
	async def fit (self) -> WhaleRadar:
		# Get default bf params
		default_radar = await self.__get_default_radar(dbs=self.dbs)

		# Data Parameter
		self.period_wmf = int(default_radar['default_bf_period_wmf']) if self.period_wmf is None else None
		self.period_wpricecorrel = int(default_radar['default_bf_period_wpricecorrel']) if self.period_wpricecorrel is None else None
		self.bar_range = max(self.period_wmf,self.period_wpricecorrel) if self.startdate is None else None
		self.enddate = datetime.date.today() if self.enddate is None else self.enddate
		
		self.default_months_range = int(default_radar['default_months_range']) if self.default_months_range is None else self.default_months_range
		if self.startdate is None:
			self.default_months_range = int((self.default_months_range/2) + int(self.bar_range/20) + (self.bar_range % 20 > 0))
		else:
			self.default_months_range = int((self.default_months_range/2) + int((self.enddate-self.startdate).days/20) + ((self.enddate-self.startdate).days % 20 >0))

		self.screener_min_value = int(default_radar['default_screener_min_value']) if self.screener_min_value is None else self.screener_min_value
		self.screener_min_frequency = int(default_radar['default_screener_min_frequency']) if self.screener_min_frequency is None else self.screener_min_frequency

		self.radar_period = int(default_radar['default_radar_period']) if self.startdate is None else None
		self.training_start_index = (int(default_radar['default_bf_training_start_index'])-50)/(100/2) if self.training_start_index is None else self.training_start_index/100
		self.training_end_index = (int(default_radar['default_bf_training_end_index'])-50)/(100/2) if self.training_end_index is None else self.training_end_index/100
		self.min_n_cluster = int(default_radar['default_bf_min_n_cluster']) if self.min_n_cluster is None else self.min_n_cluster
		self.max_n_cluster = int(default_radar['default_bf_max_n_cluster']) if self.max_n_cluster is None else self.max_n_cluster
		self.splitted_min_n_cluster = int(default_radar['default_bf_splitted_min_n_cluster']) if self.splitted_min_n_cluster is None else self.splitted_min_n_cluster
		self.splitted_max_n_cluster = int(default_radar['default_bf_splitted_max_n_cluster']) if self.splitted_max_n_cluster is None else self.splitted_max_n_cluster
		self.stepup_n_cluster_threshold = int(default_radar['default_bf_stepup_n_cluster_threshold'])/100 if self.stepup_n_cluster_threshold is None else self.stepup_n_cluster_threshold/100
		self.filter_opt_corr = int(default_radar['default_radar_filter_opt_corr'])/100 if self.filter_opt_corr is None else self.filter_opt_corr/100

		# Get Filtered StockCodes
		self.filtered_stockcodes = await self.__get_stockcodes(
			screener_min_value=self.screener_min_value,
			screener_min_frequency=self.screener_min_frequency,
			stockcode_excludes=self.stockcode_excludes,
			dbs=self.dbs)

		# Get raw data
		raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval = \
			await self.__get_stock_raw_data(
				filtered_stockcodes=self.filtered_stockcodes,
				enddate=self.enddate,
				default_months_range=self.default_months_range,
				dbs=self.dbs
				)
		
		# Get broker flow parameters for each stock in filtered_stockcodes
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
			)
		
		# Filter code based on self.optimum_corr should be greater than self.filter_opt_corr
		self.filtered_stockcodes = \
			self.filtered_stockcodes[\
				(abs(self.optimum_corr['optimum_corr']) > self.filter_opt_corr).reset_index(drop=True)\
			].reset_index(drop=True)
		raw_data_full = \
			raw_data_full[raw_data_full.index.get_level_values(0).isin(self.filtered_stockcodes)]
		raw_data_broker_nvol = \
			raw_data_broker_nvol[raw_data_broker_nvol.index.get_level_values(0).isin(self.filtered_stockcodes)]
		raw_data_broker_nval = \
			raw_data_broker_nval[raw_data_broker_nval.index.get_level_values(0).isin(self.filtered_stockcodes)]

		# Update Date Based on Data Queried
		self.enddate = raw_data_full.index.get_level_values("date").date.max()
		if self.startdate is None:
			self.startdate = raw_data_full.groupby("code").tail(self.radar_period).index.get_level_values("date").date.min()
			# Get only self.radar_period rows from last row for each group by code from raw_data_full
			raw_data_full = raw_data_full.groupby("code").tail(self.radar_period)
			raw_data_broker_nvol = raw_data_broker_nvol.groupby("code").tail(self.radar_period)
			raw_data_broker_nval = raw_data_broker_nval.groupby("code").tail(self.radar_period)
		else:
			# Get rows from self.startdate until self.enddate from raw_data_full in the first level of pandas index
			raw_data_full = raw_data_full.query("date >= @self.startdate and date <= @self.enddate")
			raw_data_broker_nvol = raw_data_broker_nvol.query("date >= @self.startdate and date <= @self.enddate")
			raw_data_broker_nval = raw_data_broker_nval.query("date >= @self.startdate and date <= @self.enddate")
		
		# Get Whale Radar Indicators
		self.radar_indicators = await self.calc_radar_indicators(
			raw_data_full = raw_data_full,
			raw_data_broker_nvol = raw_data_broker_nvol,
			raw_data_broker_nval = raw_data_broker_nval,
			selected_broker = self.selected_broker,
			y_axis_type=self.y_axis_type,
			)
		
		return self
	
	async def __get_default_radar(self, dbs:db.Session | None = next(db.get_dbs())) -> pd.Series:
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter(
				(db.DataParam.param.like("default_months_range")) | \
				(db.DataParam.param.like("default_radar_%")) | \
				(db.DataParam.param.like("default_screener_%")) | \
				(db.DataParam.param.like("default_bf_%"))
				)
		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])
	
	async def __get_stockcodes(self,
		screener_min_value: int | None = 5000000000,
		screener_min_frequency: int | None = 1000,
		stockcode_excludes: set[str] | None = set(),
		dbs: db.Session | None = next(db.get_dbs())
		) -> pd.Series:
		"""
		Get filtered stockcodes
		Filtered by:value>screener_min_value, 
					frequency>screener_min_frequency 
					stockcode_excludes
		"""
		# Query Definition
		stockcode_excludes_lower = set(x.lower() for x in stockcode_excludes)
		qry = dbs.query(db.ListStock.code)\
			.filter((db.ListStock.value > screener_min_value) &
					(db.ListStock.frequency > screener_min_frequency) &
					(db.ListStock.code.not_in(stockcode_excludes_lower)))
		
		# Query Fetching: filtered_stockcodes
		return pd.Series(pd.read_sql(sql=qry.statement,con=dbs.bind).reset_index(drop=True)['code'])
	
	# Get Net Val Sum Val Broker Transaction
	async def __get_nvsv_broker_transaction(self,
		raw_data_broker_full: pd.DataFrame
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		# Calculate nval (bval-sval) and sumval (bval+sval)
		raw_data_broker_full["nvol"] = raw_data_broker_full["bvol"] - raw_data_broker_full["svol"]
		raw_data_broker_full["nval"] = raw_data_broker_full["bval"] - raw_data_broker_full["sval"]
		raw_data_broker_full["sumval"] = raw_data_broker_full["bval"] + raw_data_broker_full["sval"]
		# Aggretate by broker then broker to column for each net and sum
		raw_data_broker_nvol = raw_data_broker_full.pivot(index=None,columns="broker",values="nvol")
		raw_data_broker_nval = raw_data_broker_full.pivot(index=None,columns="broker",values="nval")
		raw_data_broker_sumval = raw_data_broker_full.pivot(index=None,columns="broker",values="sumval")

		# Fill na
		raw_data_broker_nvol.fillna(value=0, inplace=True)
		raw_data_broker_nval.fillna(value=0, inplace=True)
		raw_data_broker_sumval.fillna(value=0, inplace=True)

		# Return
		return raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval

	async def __get_full_broker_transaction(self,
		filtered_stockcodes: pd.Series = ...,
		enddate: datetime.date = datetime.date.today(),
		default_months_range: int | None = 12,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> pd.DataFrame:
		start_qry = enddate - relativedelta(months=default_months_range)

		# Query Definition
		qry = dbs.query(
			db.StockTransaction.date,
			db.StockTransaction.code,
			db.StockTransaction.broker,
			db.StockTransaction.bvol,
			db.StockTransaction.svol,
			db.StockTransaction.bval,
			db.StockTransaction.sval
		).filter(db.StockTransaction.code.in_(filtered_stockcodes.to_list()))\
		.filter(db.StockTransaction.date.between(start_qry, enddate))

		# Main Query Fetching
		raw_data_broker_full = pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by=["code","date","broker"], ascending=[True,True,True])\
			.reset_index(drop=True).set_index(["code","date"])

		# Data Cleansing: fillna
		raw_data_broker_full.fillna(value=0, inplace=True)

		return raw_data_broker_full
	
	async def __get_stock_price_data(self,
		filtered_stockcodes: pd.Series = ...,
		enddate: datetime.date = datetime.date.today(),
		default_months_range: int | None = 12,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> pd.DataFrame:
		
		start_qry = enddate - relativedelta(months=default_months_range)

		# Query Definition
		qry = dbs.query(
			db.StockData.code,
			db.StockData.date,
			db.StockData.close,
		).filter(db.StockData.code.in_(filtered_stockcodes.to_list()))\
		.filter(db.StockData.date.between(start_qry, enddate))

		# Main Query Fetching
		raw_data_full = pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by=["code","date"],ascending=[True,True])\
			.reset_index(drop=True).set_index(["code","date"])

		# End of Method: Return or Assign Attribute
		return raw_data_full

	async def __get_stock_raw_data(self,
		filtered_stockcodes: pd.Series = ...,
		enddate: datetime.date = ...,
		default_months_range: int | None = 6,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		# Get Stockdata Full
		raw_data_full = await self.__get_stock_price_data(
			filtered_stockcodes=filtered_stockcodes,
			enddate=enddate,
			default_months_range=default_months_range,
			dbs=dbs
			)

		# Get Raw Data Broker Full
		raw_data_broker_full = await self.__get_full_broker_transaction(
			filtered_stockcodes=filtered_stockcodes,
			enddate=enddate,
			default_months_range=default_months_range,
			dbs=dbs
			)
		
		# Transform Raw Data Broker
		raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval = \
			await self.__get_nvsv_broker_transaction(raw_data_broker_full=raw_data_broker_full)

		return raw_data_full, raw_data_broker_nvol, raw_data_broker_nval, raw_data_broker_sumval
	
	async def __get_selected_broker(self,
		clustered_features: pd.DataFrame,
		centroids_cluster: pd.DataFrame,
		n_selected_cluster: int | None = 1,
		) -> list[str]:
		# Get index of max value in column 0 in centroid
		selected_cluster = (centroids_cluster[0]).nlargest(n_selected_cluster).index.tolist()
		# selected_cluster = (abs(centroids_cluster[0]*centroids_cluster[1])).nlargest(n_selected_cluster).index.tolist()
		
		# Get sorted selected broker
		selected_broker = clustered_features.loc[clustered_features["cluster"].isin(selected_cluster), :]\
			.sort_values(by="corr_ncum_close", ascending=False)\
			.index.tolist()

		return selected_broker

	async def __get_selected_broker_ncum(self,
		selected_broker: list[str],
		broker_ncum: pd.DataFrame,
		) -> pd.Series:
		# Get selected broker transaction by columns of net_stockdatatransaction, then sum each column to aggregate to date
		selected_broker_ncum = broker_ncum[selected_broker].sum(axis=1)
		# Get cumulative
		selected_broker_ncum = selected_broker_ncum.cumsum().rename("selected_broker_ncum")
		return selected_broker_ncum

	async def __get_corr_selected_broker_ncum(self,
		clustered_features: pd.DataFrame,
		raw_data_close: pd.Series,
		broker_ncum: pd.Series,
		centroids_cluster: pd.DataFrame,
		n_selected_cluster: int | None = None,
		) -> pd.Series:
		selected_broker = await self.__get_selected_broker(
			clustered_features=clustered_features,
			centroids_cluster=centroids_cluster,
			n_selected_cluster=n_selected_cluster
			)

		# Get selected broker transaction by columns of net_stockdatatransaction, then sum each column to aggregate to date
		selected_broker_ncum = await self.__get_selected_broker_ncum(selected_broker, broker_ncum)

		# Return correlation between close and selected_broker_ncum
		return selected_broker_ncum.corr(raw_data_close)

	async def __optimize_selected_cluster(self,
		clustered_features: pd.DataFrame,
		raw_data_close: pd.Series,
		broker_ncum: pd.Series,
		centroids_cluster: pd.DataFrame,
		stepup_n_cluster_threshold: float | None = 0.05,
		n_selected_cluster: int | None = None,
		) -> tuple[list[str], int, float]:
		# Check does n_selected_cluster already defined
		if n_selected_cluster is None:
			# Define correlation param
			corr_list = []

			# Iterate optimum n_cluster
			for n_selected_cluster in range(1,len(centroids_cluster)+1):
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
			index_max_corr: int = np.argmax(corr_list)
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
		min_n_cluster:int | None = 4,
		max_n_cluster:int | None = 10,
		) -> tuple[pd.DataFrame, pd.DataFrame]:

		# Get X and Y
		X = features[[x,y]].values
		# Define silhouette param
		silhouette_coefficient = []
		max_n_cluster = min(max_n_cluster, len(X)-1)

		# Iterate optimum n_cluster
		for n_cluster in range(min_n_cluster, max_n_cluster+1):
			# Clustering
			kmeans = KMeans(init="k-means++", n_clusters=n_cluster, random_state=0).fit(X)
			score = silhouette_score(X, kmeans.labels_)
			silhouette_coefficient.append(score)
		# Define optimum n_cluster
		optimum_n_cluster = np.argmax(silhouette_coefficient) + min_n_cluster

		# Clustering with optimum n cluster
		kmeans = KMeans(init="k-means++", n_clusters=optimum_n_cluster, random_state=0).fit(X)
		# Get cluster label
		features["cluster"] = kmeans.labels_
		# Get location of cluster center
		centroids_cluster = pd.DataFrame(kmeans.cluster_centers_)

		return features, centroids_cluster

	async def __get_bf_parameters(self,
		raw_data_close: pd.Series,
		raw_data_broker_nval: pd.DataFrame,
		raw_data_broker_sumval: pd.DataFrame,
		n_selected_cluster: int | None = None,
		training_start_index: float | None = 0.5,
		training_end_index: float | None = 0.75,
		splitted_min_n_cluster: int | None = 2,
		splitted_max_n_cluster: int | None = 5,
		stepup_n_cluster_threshold: float | None = 0.05,
		) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		# Only get third quartile of raw_data so not over-fitting
		length = raw_data_close.groupby(level=['code']).size()
		start_index = (length*training_start_index).astype('int')
		end_index = (length*training_end_index).astype('int')
		raw_data_close = raw_data_close.groupby(level=['code'], group_keys=False)\
			.apply(lambda x: x.iloc[start_index.loc[x.name]:end_index.loc[x.name]])
		raw_data_broker_nval = raw_data_broker_nval.groupby(level=['code'], group_keys=False)\
			.apply(lambda x: x.iloc[start_index.loc[x.name]:end_index.loc[x.name]])
		raw_data_broker_sumval = raw_data_broker_sumval.groupby(level=['code'], group_keys=False)\
			.apply(lambda x: x.iloc[start_index.loc[x.name]:end_index.loc[x.name]])
		
		# Only get raw_data_broker_nval groupby level code that doesn' all zero
		nval_true = raw_data_broker_nval.groupby(level=['code'], group_keys=False)\
			.apply(lambda x: (x!=0).any().any())
		sumval_true = raw_data_broker_sumval.groupby(level=['code'], group_keys=False)\
			.apply(lambda x: (x!=0).any().any())
		transaction_true = nval_true & sumval_true
		raw_data_close = raw_data_close.loc[transaction_true.index[transaction_true]]
		raw_data_broker_nval = raw_data_broker_nval.loc[transaction_true.index[transaction_true]]
		raw_data_broker_sumval = raw_data_broker_sumval.loc[transaction_true.index[transaction_true]]

		# Cumulate value for nval
		broker_ncum = raw_data_broker_nval.groupby(level=['code']).cumsum(axis=0)
		# Get correlation between raw_data_ncum and close
		corr_ncum_close = broker_ncum.groupby(level=['code']).corrwith(raw_data_close,axis=0)

		# Get each broker's sum of transaction value
		broker_sumval = raw_data_broker_sumval.groupby(level=['code']).sum()

		# fillna
		corr_ncum_close.fillna(value=0, inplace=True)
		broker_sumval.fillna(value=0, inplace=True)
		
		# Create broker features from corr_ncum_close and broker_sumval
		corr_ncum_close = corr_ncum_close.unstack().swaplevel(0,1).sort_index(level=0).rename('corr_ncum_close')
		broker_sumval = broker_sumval.unstack().swaplevel(0,1).sort_index(level=0).rename('broker_sumval')
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
					features=broker_features_std_pos.loc[code],
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
					features=broker_features_std_neg.loc[code],
					x='corr_ncum_close',
					y='broker_sumval',
					min_n_cluster=splitted_min_n_cluster,
					max_n_cluster=splitted_max_n_cluster,
				)
			features['code'] = code
			features = features.set_index('code', append=True).swaplevel(0,1).sort_index(level=0)
			if code in broker_features_pos.index.get_level_values('code'):
				features = features + (broker_features_pos.loc[(code),"cluster"].max()) + 1
			broker_features_neg = pd.concat([broker_features_neg, features], axis=0)

			centroids['code'] = code
			if code in broker_features_pos.index.get_level_values('code'):
				centroids.index = centroids.index + centroids_pos.loc[(code)].index.max() + 1
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
			selected_broker_code, optimum_n_selected_cluster_code, optimum_corr_code = \
				await self.__optimize_selected_cluster(
					clustered_features=broker_features.loc[(code)],
					raw_data_close=raw_data_close.loc[(code)],
					broker_ncum=broker_ncum.loc[(code)],
					centroids_cluster=broker_features_centroids.loc[(code)],
					n_selected_cluster=n_selected_cluster,
					stepup_n_cluster_threshold=stepup_n_cluster_threshold
				)
			selected_broker[code] = selected_broker_code
			optimum_n_selected_cluster[code] = optimum_n_selected_cluster_code
			optimum_corr[code] = optimum_corr_code
		
		optimum_n_selected_cluster = pd.DataFrame.from_dict(optimum_n_selected_cluster, orient='index').rename(columns={0:'optimum_n_selected_cluster'})
		optimum_corr = pd.DataFrame.from_dict(optimum_corr, orient='index').rename(columns={0:'optimum_corr'})

		return selected_broker, optimum_n_selected_cluster, optimum_corr, broker_features

	async def sum_selected_broker_transaction(self,
		raw_data_broker_nvol: pd.DataFrame,
		raw_data_broker_nval: pd.DataFrame,
		selected_broker: list[str],
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		
		selected_broker_nvol = pd.DataFrame()
		selected_broker_nval = pd.DataFrame()
		for code in raw_data_broker_nvol.index.get_level_values('code').unique():
			broker_nvol = raw_data_broker_nvol.loc[(code),selected_broker[code]].sum(axis=1)
			broker_nval = raw_data_broker_nval.loc[(code),selected_broker[code]].sum(axis=1)
			
			# Concatenate
			broker_nvol = pd.DataFrame(broker_nvol)
			broker_nvol.columns = ['broker_nvol']
			broker_nvol['code'] = code
			broker_nvol = broker_nvol.set_index('code', append=True).swaplevel(0,1).sort_index(level=0)
			selected_broker_nvol = pd.concat([selected_broker_nvol, broker_nvol], axis=0)

			broker_nval = pd.DataFrame(broker_nval)
			broker_nval.columns = ['broker_nval']
			broker_nval['code'] = code
			broker_nval = broker_nval.set_index('code', append=True).swaplevel(0,1).sort_index(level=0)
			selected_broker_nval = pd.concat([selected_broker_nval, broker_nval], axis=0)

		return selected_broker_nvol, selected_broker_nval
			
	async def calc_radar_indicators(self,
		raw_data_full: pd.DataFrame,
		raw_data_broker_nvol: pd.DataFrame,
		raw_data_broker_nval: pd.DataFrame,
		selected_broker: dict,
		y_axis_type: dp.ListRadarType | None = dp.ListRadarType.correlation,
		) -> pd.DataFrame:
		# Data Preparation
		selected_broker_nvol, selected_broker_nval = \
			await self.sum_selected_broker_transaction(
				raw_data_broker_nvol=raw_data_broker_nvol,
				raw_data_broker_nval=raw_data_broker_nval,
				selected_broker=selected_broker,
			)

		radar_indicators = pd.DataFrame()
		
		# Y Axis: WMF
		radar_indicators["mf"] = selected_broker_nval.groupby("code").sum()

		# X Axis:
		if y_axis_type.value == "correlation":
			selected_broker_nvol_cumsum = selected_broker_nvol.groupby(level='code').cumsum()
			radar_indicators[y_axis_type.value] = selected_broker_nvol_cumsum.groupby('code')\
				.corrwith(raw_data_full['close'],axis=0)
		elif y_axis_type.value == "changepercentage":
			radar_indicators[y_axis_type.value] = \
				(raw_data_full.groupby('code')['close'].nth([-1]) \
				-raw_data_full.groupby('code')['close'].nth([0])) \
				/raw_data_full.groupby('code')['close'].nth([0])
		else:
			raise Exception("Not a valid radar type")

		return radar_indicators

	async def chart(self, media_type: dp.ListMediaType | None = None):
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
