import datetime
import pandas as pd
import numpy as np
import json
import textwrap
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dependencies as dp

def html_wrap(text:str, width:int = 16, n_lines:int = 2):
	text_arr = textwrap.wrap(text=text, width=width)
	text_arr = text_arr[0:n_lines]
	if len(text_arr) >= n_lines:
		text_arr[n_lines-1] = str(text_arr[n_lines-1]) + "..."
	return "<br>".join(text_arr)

async def quantist_stock_chart(
	stockcode:str = ..., 
	wf_indicators:pd.DataFrame = ...,
	analysis_method: dp.AnalysisMethod = dp.AnalysisMethod.broker,
	period_prop: int | None = None,
	period_pricecorrel: int | None = None,
	period_mapricecorrel: int | None = None,
	period_vwap:int | None = None,
	holding_composition: pd.DataFrame | None = None,
	selected_broker: list[str] | None = None,
	optimum_n_selected_cluster: int | None = None,
	optimum_corr: float | None = None,
	) -> go.Figure:
	if analysis_method == dp.AnalysisMethod.foreign:
		abv = "F"
		method = "Foreign"
	else:
		abv = "W"
		method = "Whale"
	
	if optimum_corr is None:
		optimum_corr = np.nan
		
	# Make Subplots
	if holding_composition is not None:
		fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
			specs=[[{"secondary_y":True},{"secondary_y":False}],[{"secondary_y":True},{"secondary_y":False}]],
			vertical_spacing=0, horizontal_spacing=0.05,
			row_heights=[0.7,0.3],
			column_widths=[0.85,0.15],
			subplot_titles=("","Holding Composition","",""))
	else:
		fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
			specs=[[{"secondary_y":True}],[{"secondary_y":True}]],
			vertical_spacing=0,
			row_heights=[0.7,0.3])

	# Add Trace
	# OHLC Candlestick
	fig.add_trace(go.Candlestick(
		x=wf_indicators.index,
		open=wf_indicators["openprice"],
		high=wf_indicators["high"],
		low=wf_indicators["low"],
		close=wf_indicators["close"],
		name="Price",
		legendrank=1
		),
		row=1, col=1, secondary_y=True
	)
	# Whale VWAP
	fig.add_trace(go.Scatter(x=wf_indicators.index,y=wf_indicators["vwap"],
		name=f"{abv}-VWAP",marker_color="red",
		legendrank=2
		),
		row=1,col=1,secondary_y=True
	)
	# Whale Flow
	fig.add_trace(go.Scatter(x=wf_indicators.index,y=wf_indicators["volflow"],
		name=f"{abv} Vol Flow",marker_color="orange",
		legendrank=3
		),
		row=1,col=1,secondary_y=False
	)

	# Whale Proportion
	fig.add_trace(go.Scatter(x=wf_indicators.index,y=wf_indicators['prop']*100,
		name=f"{abv} Proportion %",marker_color="blue",
		legendrank=4
		),
		row=2,col=1,secondary_y=True
	)
	# Whale Net Proportion
	fig.add_trace(go.Scatter(x=wf_indicators.index,y=wf_indicators['netprop']*100,
		name=f"{abv} Net Proportion %",marker_color="green",
		legendrank=5
		),
		row=2,col=1,secondary_y=True
	)

	# Whale Net Value
	fig.add_trace(go.Bar(x=wf_indicators.index,y=wf_indicators["mf"],
		name=f"{abv} Net Value",marker_color=np.where(wf_indicators["mf"]<0,"red","green"),
		legendrank=6
		),
		row=2,col=1,secondary_y=False,
	)
	
	# Holding Composition
	if holding_composition is not None:
		# Append scrpless_ratio*100 to month_list text separate by |
		month_list = [f"{holding_composition.loc[x,'scripless_ratio']*100:.2f} | {x.strftime('%b')}-{x.strftime('%y')}" for x in holding_composition.index] # type: ignore
		for col_name in holding_composition.drop(labels=['scripless_ratio'],axis=1).columns:
			fig.add_trace(go.Bar(
				name=col_name,
				x=month_list,y=holding_composition[col_name]*100,
				legendrank=9,),
				row=1,col=2,secondary_y=False,
			)
		fig.update_layout(barmode='stack', xaxis2_tickangle=90)

	# Whale Volume Profile
	assert isinstance(fig.data, tuple)
	vol_profile_index = len(fig.data)
	fig.add_trace(go.Histogram(
		x=wf_indicators['netvol'],
		y=wf_indicators['close'],
		histfunc="sum",
		name="Net Volume Profile",orientation="h",opacity=0.1,
		legendrank=8),
		row=1,col=1,secondary_y=True)
	bins=np.arange(fig.full_figure_for_development(warn=False).data[vol_profile_index].ybins.start, # type:ignore
		fig.full_figure_for_development(warn=False).data[vol_profile_index].ybins.end, # type:ignore
		fig.full_figure_for_development(warn=False).data[vol_profile_index].ybins.size) # type:ignore
	bins = pd.Series(bins)
	hist_bar = wf_indicators.groupby(pd.cut(wf_indicators['close'].to_numpy(),bins=bins))['netvol'].sum() # type:ignore
	fig.data[vol_profile_index].update(marker=dict(color=np.where(hist_bar<0,"tomato","cyan")),xaxis=f'x{vol_profile_index}') # type:ignore

	# UPDATE AXES
	# Column 1
	# Row 1
	fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=True, showgrid=True)
	fig.update_yaxes(row=1, col=1,secondary_y=False,showgrid=False, zeroline=False)
	# Row 2
	fig.update_yaxes(title_text=f"{abv} Proportion %", row=2, col=1, secondary_y=True, showgrid=True, range=[0,101])
	fig.update_yaxes(title_text=f"{abv} Net Value", row=2, col=1,secondary_y=False, showgrid=False, zeroline=False)
	# Column 2
	# Row 1
	fig.update_yaxes(title_text="Ratio %", row=1, col=2, secondary_y=False, showgrid=True, side='right')

	start_temp = wf_indicators.index[0]
	end_temp = wf_indicators.index[-1]
	assert isinstance(start_temp, datetime.datetime)
	assert isinstance(end_temp, datetime.datetime)
	dt_all = pd.date_range(start=start_temp,end=end_temp)
	dt_obs = [d.strftime("%Y-%m-%d") for d in wf_indicators.index] # type: ignore
	dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]

	fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True)
	fig.update_xaxes(title_text="Scripless % | Date", row=1, col=2, showgrid=True, showticklabels=True)
	fig.update_xaxes(rangeslider={"autorange":True, "visible":False})
	fig.update_xaxes(col=1, rangebreaks=[dict(values=dt_breaks)])
	# fig.update_layout({"xaxis_range":[start_temp,end_temp+datetime.timedelta(days=round(len(wf_indicators)*0.1))]})
	# fig.update_layout({"xaxis2_range":None})

	fig.update_layout({f"xaxis{vol_profile_index}":{'anchor': 'y', 'overlaying': 'x','showgrid':False,"visible":False}})

	# ANNOTATION
	pricecorrel = wf_indicators.loc[end_temp,'pricecorrel'] # type:ignore
	if pricecorrel >= 0.7:
		pricecorrel_color = "SpringGreen"
	elif pricecorrel >= 0.4:
		pricecorrel_color = "Yellow"
	else:
		pricecorrel_color = "Red"
	pricecorrel = "{:.2f}%".format(pricecorrel*100)

	mapricecorrel = wf_indicators.loc[wf_indicators.index[-1],'mapricecorrel'] # type:ignore
	if mapricecorrel >= 0.7:
		mapricecorrel_color = "SpringGreen"
	elif mapricecorrel >= 0.4:
		mapricecorrel_color = "Yellow"
	else:
		mapricecorrel_color = "Red"
	mapricecorrel = "{:.2f}%".format(mapricecorrel*100)
	
	pow = wf_indicators.loc[wf_indicators.index[-1],'pow'] # type:ignore
	if pow == 3:
		pow_text = "<span style='color:SpringGreen'>High</span>"
	elif pow == 2:
		pow_text = "<span style='color:Yellow'>Medium</span>"
	else:
		pow_text = "<span style='color:Red'>Low</span>"
	
	fig.add_annotation(xref="x domain",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1,
		text = f"<b>Date: {end_temp.strftime('%Y-%m-%d')}</b> \
			<b>Close</b>: {'{:0.0f}'.format(wf_indicators.close[-1])}\
			<b>{abv}-VWAP({period_vwap if period_vwap is not None else ''})</b>: {'{:0.0f}'.format(wf_indicators.vwap[-1])}\
			<b>{abv}-Prop({period_prop if period_prop is not None else ''})</b>: {'{:.2f}%'.format(wf_indicators.prop[-1]*100)}\
			<b>{abv}-NetProp({period_prop if period_prop is not None else ''})</b>: {'{:.2f}%'.format(wf_indicators.netprop[-1]*100)}\
			<br><b>{abv}-Corr({period_pricecorrel if period_pricecorrel is not None else ''})</b>: <span style='color:{pricecorrel_color}'>{pricecorrel}</span>\
			<b>MA {abv}-Corr({period_mapricecorrel if period_mapricecorrel is not None else ''})</b>: <span style='color:{mapricecorrel_color}'>{mapricecorrel}</span>\
			<b>{abv}-Power</b>: {pow_text}",
		font=dict(),align="left",
		showarrow=False
	)
	
	fig.add_annotation(xref="x domain",yref="paper",xanchor="right",yanchor="bottom",x=1,y=1,
		text=f"<b>Method: <span style='color:Fuchsia'>{method} Flow</span></b> | <b>🔦 Chart by Quantist.io</b>",
		font=dict(),align="right",
		showarrow=False
	)

	if selected_broker is not None:
		fig.add_annotation(xref="x2 domain",yref="paper",xanchor="right",yanchor="bottom",x=1,y=-0.05,
			text=f"Clustering Info<br>N: {optimum_n_selected_cluster} | Corr: {'{:.2f}%'.format(optimum_corr*100)}<br>Selected Broker: {html_wrap(str(selected_broker),28,2)}",
			align="right",
			showarrow=False
		)

	# TITLE
	STOCKCODE = stockcode.upper()
	fig.update_layout(title={"text":f"<b>{STOCKCODE}</b>", "x":0.5})

	# UPDATE_LAYOUT GLOBAL DEFAULT TEMPLATE
	fig.update_layout(legend={"orientation":"h","y":-0.075, "traceorder":"normal"})
	fig.update_layout(template="plotly_dark",paper_bgcolor="#121212",plot_bgcolor="#121212")
	fig.update_layout(dragmode="pan")
	fig.update_layout(margin=dict(l=0,r=0,b=50,t=75,pad=0))

	return fig

async def radar_chart(
	startdate:datetime.date,enddate:datetime.date,
	y_axis_type:dp.ListRadarType = dp.ListRadarType.correlation,
	method:str = "Foreign",
	radar_indicators:pd.DataFrame=...
	) -> go.Figure:
	# INIT
	fig = go.Figure()

	# ADD TRACE
	fig.add_trace(go.Scatter(
		x=radar_indicators["mf"],
		y=radar_indicators[y_axis_type]*100,
		text=radar_indicators.index.str.upper(),
		textposition="bottom center",
		mode="markers+text",
		name="Whale Radar",
		marker_color="#BB86FC"
		))
	
	# UPDATE AXES
	fig.update_yaxes(title_text=y_axis_type.capitalize(), showgrid=True, zerolinewidth=3)
	fig.update_xaxes(title_text="Money Flow",showgrid=True,zerolinewidth=3)
	fig.update_xaxes(rangeslider={"autorange":True,"visible":False})
	
	# ANNOTATION
	fig.add_annotation(xref="paper",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1,
		text=f"<b>🔦 Chart by Quantist.io</b> | <b>Method: <span style='color:#BB86FC'>{method} Flow</span></b> | Data date: {startdate.strftime('%Y-%m-%d')} - {enddate.strftime('%Y-%m-%d')}",
		textangle=0,align="left",
		showarrow=False
	)
	if y_axis_type == "changepercentage":
		q1 = "ACCUMULATION AREA"
		q2 = "MARKUP AREA"
		q3 = "DISTRIBUTION AREA"
		q4 = "MARKDOWN AREA"
	else: # if y_axis_type == "correlation":
		fig.update_yaxes(range=[-101,101])
		q1 = "ACCUMULATION AREA"
		q2 = "DISTRIBUTION AREA"
		q3 = "MARKUP AREA"
		q4 = "MARKDOWN AREA"
	fig.add_annotation(xref="x domain",yref="y domain",x=1,y=1,text=f"<b>{q1}</b>",showarrow=False,font=dict(color="#BB86FC"))
	fig.add_annotation(xref="x domain",yref="y domain",x=0,y=1,text=f"<b>{q2}</b>",showarrow=False,font=dict(color="#BB86FC"))
	fig.add_annotation(xref="x domain",yref="y domain",x=0,y=0,text=f"<b>{q3}</b>",showarrow=False,font=dict(color="#BB86FC"))
	fig.add_annotation(xref="x domain",yref="y domain",x=1,y=0,text=f"<b>{q4}</b>",showarrow=False,font=dict(color="#BB86FC"))

	# TITLE
	fig.update_layout(title={"text":f"<b>{method} Radar</b>", "x":0.5})
	
	# UPDATE_LAYOUT GLOBAL DEFAULT TEMPLATE
	fig.update_layout(legend={"orientation":"h","y":-0.1})
	fig.update_layout(template="plotly_dark",paper_bgcolor="#121212",plot_bgcolor="#121212")
	fig.update_layout(dragmode="pan")
	fig.update_layout(margin=dict(l=0,r=0,b=0,t=50,pad=0))
	
	return fig

async def broker_cluster_chart(broker_features: pd.DataFrame, code:str):
	fig = px.scatter(broker_features, 
		x='corr_ncum_close', y='broker_sumval', 
		labels={'x':'Price-Net Transaction Correlation', 'y':'Sum Transaction'},
		color='cluster', symbol='cluster', text=broker_features.index,
		title=f"{code.upper()} - Broker K Means Clustering",
		color_continuous_scale=px.colors.qualitative.G10,
		)
	fig.update_traces(marker_size=20)
	fig.update_layout(font_size=20)
	fig.update_layout(coloraxis_showscale=False)

	# UPDATE_LAYOUT GLOBAL DEFAULT TEMPLATE
	fig.update_layout(legend={"orientation":"h","y":-0.1})
	fig.update_layout(template="plotly_dark",paper_bgcolor="#121212",plot_bgcolor="#121212")

	return fig

async def fig_to_json(fig:go.Figure):
	return json.dumps(fig, cls=PlotlyJSONEncoder)

async def fig_to_image(fig:go.Figure,format:str = "jpeg"):
	# File Export:
	# fig.write_image("img.jpeg", engine="kaleido", width=1920, height=1080)
	# Bytes Export:
	return fig.to_image(format=format, engine="kaleido", scale=5, width=1280, height=720)