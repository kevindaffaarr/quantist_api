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

async def foreign_chart(stockcode:str = "", ff_indicators:pd.DataFrame=...) -> go.Figure:

	# Make Subplots
	fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
		specs=[[{"secondary_y":True}],[{"secondary_y":True}]],
		vertical_spacing=0,
		row_heights=[0.7,0.3])
	fig.update_layout(xaxis3= {'anchor': 'y', 'overlaying': 'x','showgrid':False,"visible":False})

	# Add Trace
	# OHLC Candlestick
	fig.add_trace(go.Candlestick(
		x=ff_indicators.index,
		open=ff_indicators["openprice"],
		high=ff_indicators["high"],
		low=ff_indicators["low"],
		close=ff_indicators["close"],
		name="Price"
		),
		row=1, col=1, secondary_y=True
	)
	# Foreign VWAP
	fig.add_trace(go.Scatter(x=ff_indicators.index,y=ff_indicators["vwap"],
		name="F-VWAP",marker_color="red"
		),
		row=1,col=1,secondary_y=True
	)
	# Foreign Flow
	fig.add_trace(go.Scatter(x=ff_indicators.index,y=ff_indicators["fvolflow"],
		name="F Vol Flow",marker_color="orange"
		),
		row=1,col=1,secondary_y=False
	)
	# Foreign Volume Profile
	fig.add_trace(go.Histogram(
		x=(ff_indicators['netval']/ff_indicators['close']),
		y=ff_indicators['close'],
		histfunc="sum",
		name="Net Volume Profile",
		orientation="h",opacity=0.1),
		row=1,col=1,secondary_y=True)
	bins=np.arange(fig.full_figure_for_development(warn=False).data[3].ybins.start, # type:ignore
		fig.full_figure_for_development(warn=False).data[3].ybins.end, # type:ignore
		fig.full_figure_for_development(warn=False).data[3].ybins.size) # type:ignore
	bins = pd.Series(bins)
	hist_bar = ff_indicators['netval'].groupby(pd.cut(ff_indicators['close'].to_numpy(),bins=bins)).sum() # type:ignore
	fig.data[3].update(marker=dict(color=np.where(hist_bar<0,"Tomato","cyan")),xaxis='x3') # type:ignore
	# Foreign Proportion
	fig.add_trace(go.Scatter(x=ff_indicators.index,y=ff_indicators['prop']*100,
		name="F Proportion %",marker_color="blue"
		),
		row=2,col=1,secondary_y=True
	)
	# Foreign Net Proportion
	fig.add_trace(go.Scatter(x=ff_indicators.index,y=ff_indicators['fnetprop']*100,
		name="F Net Proportion %",marker_color="green"
		),
		row=2,col=1,secondary_y=True
	)
	# Foreign Net Value
	fig.add_trace(go.Bar(x=ff_indicators.index,y=ff_indicators["mf"],
		name="F Net Value",marker_color=np.where(ff_indicators["mf"]<0,"red","green")
		),
		row=2,col=1,secondary_y=False
	)
	
	# UPDATE AXES
	# Row 1
	fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=True, showgrid=True)
	fig.update_yaxes(row=1, col=1,secondary_y=False,showgrid=False, zeroline=False)
	# Row 2
	fig.update_yaxes(title_text="F Proportion %", row=2, col=1, secondary_y=True, showgrid=True, range=[0,101])
	fig.update_yaxes(title_text="F Net Value", row=2, col=1,secondary_y=False, showgrid=False, zeroline=False)

	start_temp = ff_indicators.index[0]
	end_temp = ff_indicators.index[-1]
	assert isinstance(start_temp, datetime.date)
	assert isinstance(end_temp, datetime.date)
	dt_all = pd.date_range(start=start_temp,end=end_temp)
	dt_obs = [d.strftime("%Y-%m-%d") for d in ff_indicators.index]
	dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]

	fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True)
	fig.update_xaxes(rangeslider={"autorange":True, "visible":False})
	fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
	fig.update_layout(xaxis_range=[start_temp,end_temp+datetime.timedelta(days=round(len(ff_indicators)*0.1))])

	# ANNOTATION
	pricecorrel = ff_indicators.loc[ff_indicators.index[-1],'pricecorrel'] # type:ignore
	if pricecorrel >= 0.7:
		pricecorrel_color = "SpringGreen"
	elif pricecorrel >= 0.4:
		pricecorrel_color = "Yellow"
	else:
		pricecorrel_color = "Red"
	pricecorrel = "{:.2f}%".format(pricecorrel*100)

	mapricecorrel = ff_indicators.loc[ff_indicators.index[-1],'mapricecorrel'] # type:ignore
	if mapricecorrel >= 0.7:
		mapricecorrel_color = "SpringGreen"
	elif mapricecorrel >= 0.4:
		mapricecorrel_color = "Yellow"
	else:
		mapricecorrel_color = "Red"
	mapricecorrel = "{:.2f}%".format(mapricecorrel*100)
	
	pow = ff_indicators.loc[ff_indicators.index[-1],'pow'] # type:ignore
	if pow == 3:
		pow_text = "<span style='color:SpringGreen'>High</span>"
	elif pow == 2:
		pow_text = "<span style='color:Yellow'>Medium</span>"
	else:
		pow_text = "<span style='color:Red'>Low</span>"
	
	fig.add_annotation(xref="x domain",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1,
		text = f"<b>Date: {end_temp.strftime('%Y-%m-%d')}</b> \
			<b>Close</b>: {'{:0.0f}'.format(ff_indicators.close[-1])}\
			<b>F-VWAP</b>: {'{:0.0f}'.format(ff_indicators.vwap[-1])}\
			<b>F-Prop</b>: {'{:.2f}%'.format(ff_indicators.prop[-1]*100)}\
			<b>F-NetProp</b>: {'{:.2f}%'.format(ff_indicators.fnetprop[-1]*100)}\
			<br><b>F-Corr</b>: <span style='color:{pricecorrel_color}'>{pricecorrel}</span>\
			<b>MA F-Corr</b>: <span style='color:{mapricecorrel_color}'>{mapricecorrel}</span>\
			<b>F-Power</b>: {pow_text}",
		font=dict(),align="left",
		showarrow=False
	)

	fig.add_annotation(xref="x domain",yref="paper",xanchor="right",yanchor="bottom",x=1,y=1,
		text=f"<b>Method: <span style='color:Fuchsia'>Foreign Flow</span></b> | <b>🔦 Chart by Quantist.io</b>",
		font=dict(),align="right",
		showarrow=False
	)

	# TITLE
	STOCKCODE = stockcode.upper()
	fig.update_layout(title={"text":f"<b>{STOCKCODE}</b>", "x":0.5})

	# UPDATE_LAYOUT GLOBAL DEFAULT TEMPLATE
	fig.update_layout(legend={"orientation":"h","y":-0.1})
	fig.update_layout(template="plotly_dark",paper_bgcolor="#121212",plot_bgcolor="#121212")
	fig.update_layout(dragmode="pan")
	fig.update_layout(margin=dict(l=0,r=0,b=50,t=75,pad=0))

	return fig

async def broker_chart(
	stockcode:str = ..., 
	bf_indicators:pd.DataFrame = ...,
	selected_broker: list[str] | None = None,
	optimum_n_selected_cluster: int | None = None,
	optimum_corr: float | None = None,
	period_prop: int | None = None,
	period_pricecorrel: int | None = None,
	period_mapricecorrel: int | None = None,
	period_vwap:int | None = None,
	) -> go.Figure:
	if optimum_corr is None:
		optimum_corr = np.nan
		
	# Make Subplots
	fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
		specs=[[{"secondary_y":True}],[{"secondary_y":True}]],
		vertical_spacing=0,
		row_heights=[0.7,0.3])
	fig.update_layout(xaxis3= {'anchor': 'y', 'overlaying': 'x','showgrid':False,"visible":False})

	# Add Trace
	# OHLC Candlestick
	fig.add_trace(go.Candlestick(
		x=bf_indicators.index,
		open=bf_indicators["openprice"],
		high=bf_indicators["high"],
		low=bf_indicators["low"],
		close=bf_indicators["close"],
		name="Price"
		),
		row=1, col=1, secondary_y=True
	)
	# Whale VWAP
	fig.add_trace(go.Scatter(x=bf_indicators.index,y=bf_indicators["vwap"],
		name="W-VWAP",marker_color="red"
		),
		row=1,col=1,secondary_y=True
	)
	# Whale Flow
	fig.add_trace(go.Scatter(x=bf_indicators.index,y=bf_indicators["wvolflow"],
		name="W Vol Flow",marker_color="orange"
		),
		row=1,col=1,secondary_y=False
	)
	# Whale Volume Profile
	fig.add_trace(go.Histogram(
		x=bf_indicators['netvol'],
		y=bf_indicators['close'],
		histfunc="sum",
		name="Net Volume Profile",orientation="h",opacity=0.1),
		row=1,col=1,secondary_y=True)
	bins=np.arange(fig.full_figure_for_development(warn=False).data[3].ybins.start, # type:ignore
		fig.full_figure_for_development(warn=False).data[3].ybins.end, # type:ignore
		fig.full_figure_for_development(warn=False).data[3].ybins.size) # type:ignore
	bins = pd.Series(bins)
	hist_bar = bf_indicators.groupby(pd.cut(bf_indicators['close'].to_numpy(),bins=bins))['netvol'].sum() # type:ignore
	fig.data[3].update(marker=dict(color=np.where(hist_bar<0,"tomato","cyan")),xaxis='x3') # type:ignore

	# Whale Proportion
	fig.add_trace(go.Scatter(x=bf_indicators.index,y=bf_indicators['prop']*100,
		name="W Proportion %",marker_color="blue"
		),
		row=2,col=1,secondary_y=True
	)
	# Whale Net Proportion
	fig.add_trace(go.Scatter(x=bf_indicators.index,y=bf_indicators['wnetprop']*100,
		name="W Net Proportion %",marker_color="green"
		),
		row=2,col=1,secondary_y=True
	)

	# Whale Net Value
	fig.add_trace(go.Bar(x=bf_indicators.index,y=bf_indicators["mf"],
		name="W Net Value",marker_color=np.where(bf_indicators["mf"]<0,"red","green")
		),
		row=2,col=1,secondary_y=False
	)
	
	# UPDATE AXES
	# Row 1
	fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=True, showgrid=True)
	fig.update_yaxes(row=1, col=1,secondary_y=False,showgrid=False, zeroline=False)
	# Row 2
	fig.update_yaxes(title_text="W Proportion %", row=2, col=1, secondary_y=True, showgrid=True, range=[0,101])
	fig.update_yaxes(title_text="W Net Value", row=2, col=1,secondary_y=False, showgrid=False, zeroline=False)

	start_temp = bf_indicators.index[0]
	end_temp = bf_indicators.index[-1]
	assert isinstance(start_temp, datetime.datetime)
	assert isinstance(end_temp, datetime.datetime)
	dt_all = pd.date_range(start=start_temp,end=end_temp)
	dt_obs = [d.strftime("%Y-%m-%d") for d in bf_indicators.index]
	dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]

	fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True)
	fig.update_xaxes(rangeslider={"autorange":True, "visible":False})
	fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
	fig.update_layout(xaxis_range=[start_temp,end_temp+datetime.timedelta(days=round(len(bf_indicators)*0.1))])

	# ANNOTATION
	pricecorrel = bf_indicators.loc[end_temp,'pricecorrel'] # type:ignore
	if pricecorrel >= 0.7:
		pricecorrel_color = "SpringGreen"
	elif pricecorrel >= 0.4:
		pricecorrel_color = "Yellow"
	else:
		pricecorrel_color = "Red"
	pricecorrel = "{:.2f}%".format(pricecorrel*100)

	mapricecorrel = bf_indicators.loc[bf_indicators.index[-1],'mapricecorrel'] # type:ignore
	if mapricecorrel >= 0.7:
		mapricecorrel_color = "SpringGreen"
	elif mapricecorrel >= 0.4:
		mapricecorrel_color = "Yellow"
	else:
		mapricecorrel_color = "Red"
	mapricecorrel = "{:.2f}%".format(mapricecorrel*100)
	
	pow = bf_indicators.loc[bf_indicators.index[-1],'pow'] # type:ignore
	if pow == 3:
		pow_text = "<span style='color:SpringGreen'>High</span>"
	elif pow == 2:
		pow_text = "<span style='color:Yellow'>Medium</span>"
	else:
		pow_text = "<span style='color:Red'>Low</span>"
	
	fig.add_annotation(xref="x domain",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1,
		text = f"<b>Date: {end_temp.strftime('%Y-%m-%d')}</b> \
			<b>Close</b>: {'{:0.0f}'.format(bf_indicators.close[-1])}\
			<b>W-VWAP({period_vwap if period_vwap is not None else ''})</b>: {'{:0.0f}'.format(bf_indicators.vwap[-1])}\
			<b>W-Prop({period_prop if period_prop is not None else ''})</b>: {'{:.2f}%'.format(bf_indicators.prop[-1]*100)}\
			<b>W-NetProp({period_prop if period_prop is not None else ''})</b>: {'{:.2f}%'.format(bf_indicators.wnetprop[-1]*100)}\
			<br><b>W-Corr({period_pricecorrel if period_pricecorrel is not None else ''})</b>: <span style='color:{pricecorrel_color}'>{pricecorrel}</span>\
			<b>MA W-Corr({period_mapricecorrel if period_mapricecorrel is not None else ''})</b>: <span style='color:{mapricecorrel_color}'>{mapricecorrel}</span>\
			<b>W-Power</b>: {pow_text}",
		font=dict(),align="left",
		showarrow=False
	)
	
	fig.add_annotation(xref="x domain",yref="paper",xanchor="right",yanchor="bottom",x=1,y=1,
		text=f"<b>Method: <span style='color:Fuchsia'>Whale Flow</span></b> | <b>🔦 Chart by Quantist.io</b>",
		font=dict(),align="right",
		showarrow=False
	)

	fig.add_annotation(xref="x domain",yref="paper",xanchor="right",yanchor="bottom",x=1,y=-0.15,
		text=f"Clustering Info<br>N: {optimum_n_selected_cluster} | Corr: {'{:.2f}%'.format(optimum_corr*100)}<br>Selected Broker: {html_wrap(str(selected_broker),32,2)}",
		font=dict(),align="right",
		showarrow=False
	)

	# TITLE
	STOCKCODE = stockcode.upper()
	fig.update_layout(title={"text":f"<b>{STOCKCODE}</b>", "x":0.5})

	# UPDATE_LAYOUT GLOBAL DEFAULT TEMPLATE
	fig.update_layout(legend={"orientation":"h","y":-0.1})
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