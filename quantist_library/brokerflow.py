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
		default_year_range = int(default_bf['default_year_range']) if self.startdate is None else 0
		self.enddate = datetime.date.today() if self.enddate is None else self.enddate
		self.startdate = self.enddate - relativedelta(years=default_year_range) if self.startdate is None else self.startdate
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
				n_selected_cluster=self.n_selected_cluster
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
				(db.DataParam.param.like("default_year_range")))
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
		default_year_range: int | None = 1,
		dbs: db.Session | None = next(db.get_dbs()),
		):
		# if startdate is none, set to 1 year before enddate
		if preoffset_startdate is None:
			preoffset_startdate = enddate - relativedelta(years=default_year_range)

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

	async def __get_stock_full_data(self,
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		preoffset_period_param: int | None = 50,
		default_year_range: int | None = 1,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> pd.Series:
		# Define startdate if None equal to last year of enddate
		if startdate is None:
			startdate = enddate - relativedelta(years=default_year_range)

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
		self.preoffset_startdate = raw_data_pre.index[0].date()

		# Concatenate Pre and Main Query
		raw_data_full = pd.concat([raw_data_pre,raw_data_main])

		# Data Cleansing: zero openprice replace with previous
		raw_data_full['openprice'] = raw_data_full['openprice'].mask(raw_data_full['openprice'].eq(0),raw_data_full['previous'])

		# End of Method: Return or Assign Attribute
		return raw_data_full

	async def __get_stock_raw_data(self,
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_year_range: int | None = 1,
		preoffset_period_param: int | None = 50,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		# Get Stockdata Full
		raw_data_full = await self.__get_stock_full_data(
			stockcode=stockcode,startdate=startdate,enddate=enddate,
			preoffset_period_param=preoffset_period_param,
			default_year_range=default_year_range,dbs=dbs)

		# Get Raw Data Broker Full
		raw_data_broker_full = await self.__get_full_broker_transaction(
			stockcode=stockcode,preoffset_startdate=self.preoffset_startdate,enddate=enddate,
			default_year_range=default_year_range,dbs=dbs)

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
		selected_cluster = centroids_cluster[0].nlargest(n_selected_cluster).index.tolist()
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
		selected_broker = await self.__get_selected_broker(clustered_features=clustered_features,centroids_cluster=centroids_cluster,n_selected_cluster=n_selected_cluster)

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
				selected_broker_ncum_corr = await self.__get_corr_selected_broker_ncum(clustered_features, raw_data_close, broker_ncum, centroids_cluster, n_selected_cluster)
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

		# If n_selected_cluster not defined
		else:
			optimum_n_selected_cluster: int = n_selected_cluster
			optimum_corr = await self.__get_corr_selected_broker_ncum(clustered_features, raw_data_close, broker_ncum, centroids_cluster, n_selected_cluster)

		# Get Selected Broker from optimum n_selected_cluster
		selected_broker = await self.__get_selected_broker(clustered_features=clustered_features,centroids_cluster=centroids_cluster,n_selected_cluster=optimum_n_selected_cluster)

		return selected_broker, optimum_n_selected_cluster, optimum_corr

	async def __kmeans_clustering(self,
		features: pd.DataFrame,
		x: str,
		y: str,
		MIN_N_CLUSTER:int | None = 4,
		MAX_N_CLUSTER:int | None = 10,
		) -> tuple[pd.DataFrame, pd.DataFrame]:

		# Get X and Y
		X = features[[x,y]].values
		# Define silhouette param
		silhouette_coefficient = []

		# Iterate optimum n_cluster
		for n_cluster in range(MIN_N_CLUSTER, MAX_N_CLUSTER+1):
			# Clustering
			kmeans = KMeans(init="k-means++", n_clusters=n_cluster, random_state=0).fit(X)
			score = silhouette_score(X, kmeans.labels_)
			silhouette_coefficient.append(score)
		# Define optimum n_cluster
		optimum_n_cluster = np.argmax(silhouette_coefficient) + MIN_N_CLUSTER

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
		) -> tuple[list[str], int, float]:

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

		# Standardize Features
		broker_features_std = await self.__xy_standardize(broker_features)
		# Clustering
		broker_features_cluster, broker_features_centroids = await self.__kmeans_clustering(broker_features_std, "corr_ncum_close", "broker_sumval")
		# Get cluster label
		broker_features["cluster"] = broker_features_cluster["cluster"].astype("int")

		# Delete variable for memory management
		del broker_features_std, broker_features_cluster
		gc.collect()

		# Define optimum selected cluster: net transaction clusters with highest correlation to close
		selected_broker, optimum_n_selected_cluster, optimum_corr = \
			await self.__optimize_selected_cluster(
				clustered_features=broker_features,
				raw_data_close=raw_data_close,
				broker_ncum=broker_ncum,
				centroids_cluster=broker_features_centroids,
				n_selected_cluster=n_selected_cluster
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
		) -> tuple[list[str], int, float]:
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
	def __init__(self) -> None:
		pass
	
	async def fit (self) -> WhaleRadar:
		return self