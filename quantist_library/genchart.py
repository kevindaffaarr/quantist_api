import datetime
import pandas as pd
import numpy as np
import json
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def foreign_chart(stockcode:str | None=None,ff_indicators:pd.DataFrame = ...):
	# Make Subplots
	fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
		specs=[[{"secondary_y":True}],[{"secondary_y":True}]],
		vertical_spacing=0,
		row_heights=[0.7,0.3])
	
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
	fig.add_trace(go.Bar(x=ff_indicators.index,y=ff_indicators["netval"],
		name="F Net Value",marker_color=np.where(ff_indicators["netval"]<0,"red","green")
		),
		row=2,col=1,secondary_y=False
	)
	
	# UPDATE AXES
	# Row 1
	fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=True, showgrid=True)
	fig.update_yaxes(row=1, col=1,secondary_y=False,showgrid=False, zeroline=False)
	# Row 2
	fig.update_yaxes(title_text="F Proportion %", row=2, col=1, secondary_y=True, showgrid=True)
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
	
	fig.add_annotation(xref="paper",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1,
		text=f"<b>🔦 Chart by Quantist.io</b> | <b>Method: <span style='color:Fuchsia'>Foreign Flow</span></b> | <b>F-Power</b>: {fpow_text}",
		font=dict(),textangle=0,
		showarrow=False
	)

	fig.add_annotation(xref="paper",yref="paper",xanchor="left",yanchor="bottom",x=0.025,y=0.75,
		align="left",bordercolor="Ivory",borderwidth=0,
		showarrow=False,
		text=f"\
			<b>Date: {datetime.datetime.strftime(ff_indicators.index[-1],'%Y-%m-%d')}</b><br>\
			Close: {'{:0.0f}'.format(ff_indicators.close[-1])}<br>\
			F-VWAP: {'{:0.0f}'.format(ff_indicators.fvwap[-1])}<br>\
			F-Prop: {'{:.2f}%'.format(ff_indicators.fprop[-1]*100)}<br>\
			F-NetProp: {'{:.2f}%'.format(ff_indicators.fnetprop[-1]*100)}<br>\
			F-Corr: <span style='color:{fpricecorrel_color}'>{fpricecorrel}</span><br>\
			MA F-Corr: <span style='color:{fmapricecorrel_color}'>{fmapricecorrel}</span><br>", 
        )

	# TITLE
	STOCKCODE = stockcode.upper()
	fig.update_layout(title={"text":f"<b>{STOCKCODE}</b>", "x":0.5})
	fig.update_layout(legend={"orientation":"h","y":-0.1})

	# UPDATE_LAYOUT GLOBAL DEFAULT TEMPLATE
	fig.update_layout(template="plotly_dark")
	fig.update_layout(dragmode="pan")
	fig.update_layout(margin=dict(l=0,r=0,b=0,t=50,pad=0))

	return fig

def fig_to_json(fig:go.Figure):
	return json.dumps(fig, cls=PlotlyJSONEncoder)

def fig_to_image(fig:go.Figure,format:str | None = "jpeg"):
	# File Export:
	# fig.write_image("img.jpeg", engine="kaleido", width=1920, height=1080)
	# Bytes Export:
	return fig.to_image(format=format, engine="kaleido", width=1920, height=1080, scale=2)