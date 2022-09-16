import datetime
import pandas as pd
import numpy as np
import json
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots

async def foreign_chart(stockcode:str|None=None, ff_indicators:pd.DataFrame=...) -> go.Figure:
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
	fig.add_trace(go.Scatter(x=ff_indicators.index,y=ff_indicators["fvwap"],
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
	bins=np.arange(fig.full_figure_for_development(warn=False).data[3].ybins.start,\
		fig.full_figure_for_development(warn=False).data[3].ybins.end,\
		fig.full_figure_for_development(warn=False).data[3].ybins.size)
	hist_bar = ff_indicators.groupby(pd.cut(ff_indicators['close'],bins=bins))['netval'].sum()
	fig.data[3].update(marker=dict(color=np.where(hist_bar<0,"Tomato","cyan")),xaxis='x3')
	# Foreign Proportion
	fig.add_trace(go.Scatter(x=ff_indicators.index,y=ff_indicators['fprop']*100,
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
	fig.add_trace(go.Bar(x=ff_indicators.index,y=ff_indicators["fmf"],
		name="F Net Value",marker_color=np.where(ff_indicators["fmf"]<0,"red","green")
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

	dt_all = pd.date_range(start=ff_indicators.index[0],end=ff_indicators.index[-1])
	dt_obs = [d.strftime("%Y-%m-%d") for d in ff_indicators.index]
	dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]

	fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True)
	fig.update_xaxes(rangeslider={"autorange":True, "visible":False})
	fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
	fig.update_layout(xaxis_range=[ff_indicators.index[0],ff_indicators.index[-1]+datetime.timedelta(days=round(len(ff_indicators)*0.1))])

	# ANNOTATION
	fpricecorrel = ff_indicators.loc[ff_indicators.index[-1],'fpricecorrel']
	if fpricecorrel >= 0.7:
		fpricecorrel_color = "SpringGreen"
	elif fpricecorrel >= 0.4:
		fpricecorrel_color = "Yellow"
	else:
		fpricecorrel_color = "Red"
	fpricecorrel = "{:.2f}%".format(fpricecorrel*100)

	fmapricecorrel = ff_indicators.loc[ff_indicators.index[-1],'fmapricecorrel']
	if fmapricecorrel >= 0.7:
		fmapricecorrel_color = "SpringGreen"
	elif fmapricecorrel >= 0.4:
		fmapricecorrel_color = "Yellow"
	else:
		fmapricecorrel_color = "Red"
	fmapricecorrel = "{:.2f}%".format(fmapricecorrel*100)
	
	fpow = ff_indicators.loc[ff_indicators.index[-1],'fpow']
	if fpow == 3:
		fpow_text = "<span style='color:SpringGreen'>High</span>"
	elif fpow == 2:
		fpow_text = "<span style='color:Yellow'>Medium</span>"
	else:
		fpow_text = "<span style='color:Red'>Low</span>"
	
	fig.add_annotation(xref="x domain",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1,
		text=f"<b>Date: {datetime.datetime.strftime(ff_indicators.index[-1],'%Y-%m-%d')}</b> \
			<b>Close</b>: {'{:0.0f}'.format(ff_indicators.close[-1])}\
			<b>F-VWAP</b>: {'{:0.0f}'.format(ff_indicators.fvwap[-1])}\
			<b>F-Prop</b>: {'{:.2f}%'.format(ff_indicators.fprop[-1]*100)}\
			<b>F-NetProp</b>: {'{:.2f}%'.format(ff_indicators.fnetprop[-1]*100)}\
			<b>F-Corr</b>: <span style='color:{fpricecorrel_color}'>{fpricecorrel}</span>\
			<b>MA F-Corr</b>: <span style='color:{fmapricecorrel_color}'>{fmapricecorrel}</span>\
			<b>F-Power</b>: {fpow_text}",
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
	fig.update_layout(margin=dict(l=0,r=0,b=0,t=50,pad=0))

	return fig

async def broker_chart(stockcode:str|None=None, bf_indicators:pd.DataFrame=...) -> go.Figure:
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
	fig.add_trace(go.Scatter(x=bf_indicators.index,y=bf_indicators["wvwap"],
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
	bins=np.arange(fig.full_figure_for_development(warn=False).data[3].ybins.start,\
		fig.full_figure_for_development(warn=False).data[3].ybins.end,\
		fig.full_figure_for_development(warn=False).data[3].ybins.size)
	hist_bar = bf_indicators.groupby(pd.cut(bf_indicators['close'],bins=bins))['netvol'].sum()
	fig.data[3].update(marker=dict(color=np.where(hist_bar<0,"tomato","cyan")),xaxis='x3')

	# Whale Proportion
	fig.add_trace(go.Scatter(x=bf_indicators.index,y=bf_indicators['wprop']*100,
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
	fig.add_trace(go.Bar(x=bf_indicators.index,y=bf_indicators["wmf"],
		name="W Net Value",marker_color=np.where(bf_indicators["wmf"]<0,"red","green")
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

	dt_all = pd.date_range(start=bf_indicators.index[0],end=bf_indicators.index[-1])
	dt_obs = [d.strftime("%Y-%m-%d") for d in bf_indicators.index]
	dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]

	fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True)
	fig.update_xaxes(rangeslider={"autorange":True, "visible":False})
	fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
	fig.update_layout(xaxis_range=[bf_indicators.index[0],bf_indicators.index[-1]+datetime.timedelta(days=round(len(bf_indicators)*0.1))])

	# ANNOTATION
	wpricecorrel = bf_indicators.loc[bf_indicators.index[-1],'wpricecorrel']
	if wpricecorrel >= 0.7:
		wpricecorrel_color = "SpringGreen"
	elif wpricecorrel >= 0.4:
		wpricecorrel_color = "Yellow"
	else:
		wpricecorrel_color = "Red"
	wpricecorrel = "{:.2f}%".format(wpricecorrel*100)

	wmapricecorrel = bf_indicators.loc[bf_indicators.index[-1],'wmapricecorrel']
	if wmapricecorrel >= 0.7:
		wmapricecorrel_color = "SpringGreen"
	elif wmapricecorrel >= 0.4:
		wmapricecorrel_color = "Yellow"
	else:
		wmapricecorrel_color = "Red"
	wmapricecorrel = "{:.2f}%".format(wmapricecorrel*100)
	
	wpow = bf_indicators.loc[bf_indicators.index[-1],'wpow']
	if wpow == 3:
		wpow_text = "<span style='color:SpringGreen'>High</span>"
	elif wpow == 2:
		wpow_text = "<span style='color:Yellow'>Medium</span>"
	else:
		wpow_text = "<span style='color:Red'>Low</span>"
	
	fig.add_annotation(xref="x domain",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1,
		text=f"<b>Date: {datetime.datetime.strftime(bf_indicators.index[-1],'%Y-%m-%d')}</b> \
			<b>Close</b>: {'{:0.0f}'.format(bf_indicators.close[-1])}\
			<b>W-VWAP</b>: {'{:0.0f}'.format(bf_indicators.wvwap[-1])}\
			<b>W-Prop</b>: {'{:.2f}%'.format(bf_indicators.wprop[-1]*100)}\
			<b>W-NetProp</b>: {'{:.2f}%'.format(bf_indicators.wnetprop[-1]*100)}\
			<b>W-Corr</b>: <span style='color:{wpricecorrel_color}'>{wpricecorrel}</span>\
			<b>MA W-Corr</b>: <span style='color:{wmapricecorrel_color}'>{wmapricecorrel}</span>\
			<b>W-Power</b>: {wpow_text}",
		font=dict(),align="left",
		showarrow=False
	)

	fig.add_annotation(xref="x domain",yref="paper",xanchor="right",yanchor="bottom",x=1,y=1,
		text=f"<b>Method: <span style='color:Fuchsia'>Whale Flow</span></b> | <b>🔦 Chart by Quantist.io</b>",
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
	fig.update_layout(margin=dict(l=0,r=0,b=0,t=50,pad=0))

	return fig

async def radar_chart(startdate:datetime.date,enddate:datetime.date,
	y_axis_type:str|None="correlation",
	radar_indicators:pd.DataFrame=...
	) -> go.Figure:
	
	# INIT
	fig = go.Figure()

	# ADD TRACE
	fig.add_trace(go.Scatter(
		x=radar_indicators["fmf"],
		y=radar_indicators[y_axis_type]*100,
		text=radar_indicators.index.str.upper(),
		textposition="bottom center",
		mode="markers+text",
		name="Whale Radar",
		marker_color="#BB86FC"
		))
	
	# UPDATE AXES
	fig.update_yaxes(title_text=y_axis_type.capitalize(), showgrid=True, zerolinewidth=3)
	fig.update_xaxes(title_text="Foreign Money Flow",showgrid=True,zerolinewidth=3)
	fig.update_xaxes(rangeslider={"autorange":True,"visible":False})
	
	# ANNOTATION
	fig.add_annotation(xref="paper",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1,
		text=f"<b>🔦 Chart by Quantist.io</b> | <b>Method: <span style='color:#BB86FC'>Foreign Flow</span></b> | Data date: {datetime.datetime.strftime(startdate,'%Y-%m-%d')} - {datetime.datetime.strftime(enddate,'%Y-%m-%d')}",
		textangle=0,align="left",
		showarrow=False
	)
	if y_axis_type == "correlation":
		fig.update_yaxes(range=[-101,101])
		q1 = "ACCUMULATION AREA"
		q2 = "DISTRIBUTION AREA"
		q3 = "MARKUP AREA"
		q4 = "MARKDOWN AREA"
	elif y_axis_type == "changepercentage":
		q1 = "ACCUMULATION AREA"
		q2 = "MARKUP AREA"
		q3 = "DISTRIBUTION AREA"
		q4 = "MARKDOWN AREA"
	fig.add_annotation(xref="x domain",yref="y domain",x=1,y=1,text=f"<b>{q1}</b>",showarrow=False,font=dict(color="#BB86FC"))
	fig.add_annotation(xref="x domain",yref="y domain",x=0,y=1,text=f"<b>{q2}</b>",showarrow=False,font=dict(color="#BB86FC"))
	fig.add_annotation(xref="x domain",yref="y domain",x=0,y=0,text=f"<b>{q3}</b>",showarrow=False,font=dict(color="#BB86FC"))
	fig.add_annotation(xref="x domain",yref="y domain",x=1,y=0,text=f"<b>{q4}</b>",showarrow=False,font=dict(color="#BB86FC"))

	# TITLE
	fig.update_layout(title={"text":f"<b>Whale Radar</b>", "x":0.5})
	
	# UPDATE_LAYOUT GLOBAL DEFAULT TEMPLATE
	fig.update_layout(legend={"orientation":"h","y":-0.1})
	fig.update_layout(template="plotly_dark",paper_bgcolor="#121212",plot_bgcolor="#121212")
	fig.update_layout(dragmode="pan")
	fig.update_layout(margin=dict(l=0,r=0,b=0,t=50,pad=0))
	
	return fig

async def fig_to_json(fig:go.Figure):
	return json.dumps(fig, cls=PlotlyJSONEncoder)

async def fig_to_image(fig:go.Figure,format:str | None = "jpeg"):
	# File Export:
	# fig.write_image("img.jpeg", engine="kaleido", width=1920, height=1080)
	# Bytes Export:
	return fig.to_image(format=format, engine="kaleido", scale=5, width=1280, height=720)