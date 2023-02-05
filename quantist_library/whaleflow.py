from __future__ import annotations
import datetime
import pandas as pd

import database as db
import dependencies as dp

from quantist_library import foreignflow as ff, brokerflow as bf, holdingcomposition as hc
from quantist_library import genchart

class WhaleFlow():
	def __init__(self,
		analysis_method: dp.AnalysisMethod,
		stockcode: str|None = None,
		startdate: datetime.date|None = None,
		enddate: datetime.date = datetime.date.today(),
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
		dbs: db.Session = next(db.get_dbs())
		) -> None:
		self.analysis_method = analysis_method
		self.stockcode = stockcode
		self.startdate = startdate
		self.enddate = enddate
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
		self.dbs = dbs

		self.wf_indicators = pd.DataFrame()

	async def get_holding_composition(self,
		stockcode: str,
		startdate: datetime.date,
		enddate: datetime.date,
		categorization: dp.HoldingSectorsCat|None = dp.HoldingSectorsCat.default,
		) -> pd.DataFrame:
		# Change startdate to first day of the month, and enddate to last day of the month of enddate
		startdate = startdate.replace(day=1)
		enddate = enddate.replace(day=1) - datetime.timedelta(days=1)
		hc_obj = hc.HoldingComposition(stockcode=stockcode, startdate=startdate, enddate=enddate, categorization=categorization)
		hc_obj = await hc_obj.get()
		return hc_obj.holding_composition
	
	async def _full_data_processing(self):
		# Holding Composition
		assert self.stockcode is not None
		assert self.startdate is not None
		self.holding_composition = await self.get_holding_composition(stockcode=self.stockcode, startdate=self.startdate, enddate=self.enddate)

	async def _gen_full_chart(self,
		wf_indicators:pd.DataFrame,
		media_type: dp.ListMediaType | None = None,
		selected_broker: list[str] | None = None,
		optimum_n_selected_cluster: int | None = None,
		optimum_corr: float | None = None,
		):
		assert self.stockcode is not None
		fig = await genchart.quantist_stock_chart(
			stockcode=self.stockcode,
			wf_indicators=wf_indicators,
			analysis_method=self.analysis_method,
			period_prop=self.period_prop,
			period_pricecorrel=self.period_pricecorrel,
			period_mapricecorrel=self.period_mapricecorrel,
			period_vwap=self.period_vwap,
			holding_composition=self.holding_composition,
			selected_broker=selected_broker,
			optimum_n_selected_cluster=optimum_n_selected_cluster,
			optimum_corr=optimum_corr,
			)
		if media_type in ["png","jpeg","jpg","webp","svg"]:
			return await genchart.fig_to_image(fig,media_type)
		elif media_type == "json":
			return await genchart.fig_to_json(fig)
		else:
			return fig

class ForeignFlow(WhaleFlow):
	def __init__(self,
		stockcode: str | None = None,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
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
		dbs: db.Session = next(db.get_dbs())
		) -> None:
		WhaleFlow.__init__(self,
			analysis_method=dp.AnalysisMethod.foreign,
			stockcode=stockcode,
			startdate=startdate,
			enddate=enddate,
			period_mf=period_mf,
			period_prop=period_prop,
			period_pricecorrel=period_pricecorrel,
			period_mapricecorrel=period_mapricecorrel,
			period_vwap=period_vwap,
			pow_high_prop=pow_high_prop,
			pow_high_pricecorrel=pow_high_pricecorrel,
			pow_high_mapricecorrel=pow_high_mapricecorrel,
			pow_medium_prop=pow_medium_prop,
			pow_medium_pricecorrel=pow_medium_pricecorrel,
			pow_medium_mapricecorrel=pow_medium_mapricecorrel,
			dbs=dbs)
	
	async def __get_wf_obj(self) -> ForeignFlow:
		# ForeignFlow
		wf_obj = ff.StockFFFull(
			stockcode = self.stockcode,
			startdate = self.startdate,
			enddate = self.enddate,
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
			dbs = self.dbs,
		)
		wf_obj = await wf_obj.fit()
		# Update Attribute of ForeignFlow from wf_obj
		self.__dict__.update(vars(wf_obj))
		return self

	async def fit(self):
		await self.__get_wf_obj()
		await WhaleFlow._full_data_processing(self)
	
	async def chart(self, media_type: dp.ListMediaType | None = None):
		return await WhaleFlow._gen_full_chart(self,
			wf_indicators=self.wf_indicators,
			media_type=media_type,
			)


class BrokerFlow(WhaleFlow):
	def __init__(self,
		stockcode: str | None = None,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
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
		stepup_n_cluster_threshold: int | None = None,
		dbs: db.Session = next(db.get_dbs()),
		) -> None:
		WhaleFlow.__init__(self,
			analysis_method=dp.AnalysisMethod.broker,
			stockcode=stockcode,
			startdate=startdate,
			enddate=enddate,
			period_mf=period_mf,
			period_prop=period_prop,
			period_pricecorrel=period_pricecorrel,
			period_mapricecorrel=period_mapricecorrel,
			period_vwap=period_vwap,
			pow_high_prop=pow_high_prop,
			pow_high_pricecorrel=pow_high_pricecorrel,
			pow_high_mapricecorrel=pow_high_mapricecorrel,
			pow_medium_prop=pow_medium_prop,
			pow_medium_pricecorrel=pow_medium_pricecorrel,
			pow_medium_mapricecorrel=pow_medium_mapricecorrel,
			dbs=dbs)
		self.n_selected_cluster = n_selected_cluster
		self.training_start_index = training_start_index
		self.training_end_index = training_end_index
		self.min_n_cluster = min_n_cluster
		self.max_n_cluster = max_n_cluster
		self.splitted_min_n_cluster = splitted_min_n_cluster
		self.splitted_max_n_cluster = splitted_max_n_cluster
		self.stepup_n_cluster_threshold = stepup_n_cluster_threshold
	
	async def __get_wf_obj(self) -> BrokerFlow:
		# BrokerFlow
		wf_obj = bf.StockBFFull(
			stockcode = self.stockcode,
			startdate = self.startdate,
			enddate = self.enddate,
			n_selected_cluster = self.n_selected_cluster,
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
			training_start_index = self.training_start_index,
			training_end_index = self.training_end_index,
			min_n_cluster = self.min_n_cluster,
			max_n_cluster = self.max_n_cluster,
			splitted_min_n_cluster = self.splitted_min_n_cluster,
			splitted_max_n_cluster = self.splitted_max_n_cluster,
			stepup_n_cluster_threshold = self.stepup_n_cluster_threshold,
			dbs = self.dbs,
		)
		wf_obj = await wf_obj.fit()
		# Update Attribute of ForeignFlow from wf_obj
		self.__dict__.update(vars(wf_obj))
		return self
	
	async def fit(self):
		await self.__get_wf_obj()
		await WhaleFlow._full_data_processing(self)

	async def chart(self, media_type: dp.ListMediaType | None = None):
		return await WhaleFlow._gen_full_chart(self,
			wf_indicators=self.wf_indicators,
			media_type=media_type,
			)