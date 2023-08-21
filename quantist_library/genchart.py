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

pd.options.mode.copy_on_write = True

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
	fig.add_trace(go.Scatter(x=wf_indicators.index,y=wf_indicators["valflow"],
		name=f"{abv} Val Flow",marker_color="orange",
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

	# Whale Value Profile
	assert isinstance(fig.data, tuple)
	val_profile_index = len(fig.data)
	fig.add_trace(go.Histogram(
		x=wf_indicators['netval'],
		y=wf_indicators['close'],
		histfunc="sum",
		name="Net Value Profile",orientation="h",opacity=0.1,
		legendrank=8),
		row=1,col=1,secondary_y=True)
	bins=np.arange(fig.full_figure_for_development(warn=False).data[val_profile_index].ybins.start, # type:ignore
		fig.full_figure_for_development(warn=False).data[val_profile_index].ybins.end, # type:ignore
		fig.full_figure_for_development(warn=False).data[val_profile_index].ybins.size) # type:ignore
	bins = pd.Series(bins)
	hist_bar = wf_indicators.groupby(pd.cut(wf_indicators['close'].to_numpy(),bins=bins))['netval'].sum() # type:ignore
	fig.data[val_profile_index].update(marker=dict(color=np.where(hist_bar<0,"tomato","cyan")),xaxis=f'x{val_profile_index}') # type:ignore

	# UPDATE AXES
	# Column 1
	# Row 1
	fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=True, showgrid=True, range=[wf_indicators["low"].min()*0.95,wf_indicators["high"].max()*1.05])
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

	fig.update_layout({f"xaxis{val_profile_index}":{'anchor': 'y', 'overlaying': 'x','showgrid':False,"visible":False}})

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
		text=f"<b>Method: <span style='color:Fuchsia'>{method} Flow</span></b> | <b>Chart by Quantist.io</b>",
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
		text=f"<b>Chart by Quantist.io</b> | <b>Method: <span style='color:#BB86FC'>{method} Flow</span></b> | Data date: {startdate.strftime('%Y-%m-%d')} - {enddate.strftime('%Y-%m-%d')}",
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

async def broker_cluster_chart(broker_features: pd.DataFrame, broker_nval_last: pd.Series, code:str, startdate: datetime.date, enddate: datetime.date) -> go.Figure:
	# Make subplots with 2 columns, 1 row. Left column is scatter (70%), right column for table (30%)
	fig = make_subplots(
		rows=1, cols=2, 
		column_widths=[0.7, 0.3], horizontal_spacing=0.02,
		specs=[[{"type": "scatter"}, {"type": "table"}]],
		subplot_titles=(
			"",
			"<b>Last Broker Summary</b>"
		)
	)
	fig.update_annotations(font_size=15, align="left")

	# Add scatter trace
	fig.add_trace(
		go.Scatter(
			x=broker_features['corr_ncum_close'], y=broker_features['broker_sumval'],
			text=broker_features.index,
			mode='markers+text',
			textposition='bottom center',
			marker=dict(
				size=15,
				color=broker_features['cluster'],
				colorscale=px.colors.qualitative.G10,
				showscale=False,
				line_width=1,
				symbol=broker_features['cluster']
			),
			textfont=dict(size=20, color='white'),
		),
		row=1, col=1
	)
	# Update Axis Title
	fig.update_xaxes(title=dict(text="Price-Transaction Movement Correlation", font_size=15), tickfont=dict(size=15), row=1, col=1)
	fig.update_yaxes(title=dict(text="Total Transaction Value", font_size=15), tickfont=dict(size=15), row=1, col=1)

	# Add table trace
	# Get broker name set with maximum corr_cluster
	list_max_broker = broker_features.loc[broker_features['corr_cluster'] == broker_features['corr_cluster'].max()].index.to_list()
	# Sort broker_nval_last by positive and negative
	pos_broker_nval_last = broker_nval_last[broker_nval_last > 0].sort_values(ascending=False)
	neg_broker_nval_last = broker_nval_last[broker_nval_last < 0].sort_values(ascending=True).abs()
	# Format broker_nval_last to B, M, K with 2 decimal places
	pos_broker_nval_last = pos_broker_nval_last.apply(lambda x: f"{x/1000000000:.2f}B" if x >= 1000000000 else f"{x/1000000:.2f}M" if x >= 1000000 else f"{x/1000:.2f}K")
	neg_broker_nval_last = neg_broker_nval_last.apply(lambda x: f"{x/1000000000:.2f}B" if x >= 1000000000 else f"{x/1000000:.2f}M" if x >= 1000000 else f"{x/1000:.2f}K")

	# Fill Color is #00A08B for 1st and 2nd column, #AF0038 for 3rd and 4th column, and #9467BD specially if broker is in list_max_broker
	fig.add_trace(
		go.Table(
			header=dict(
				values=["<b>Broker</b>", "<b>Net Buy</b>", "<b>Broker</b>", "<b>Net Sell</b>"],
				font_size=15,
				align="center"
			),
			cells=dict(
				values=[
					pos_broker_nval_last.index,
					pos_broker_nval_last.values,
					neg_broker_nval_last.index,
					neg_broker_nval_last.values,
				],
				fill_color=[
					["#9467BD" if x in list_max_broker else "#00A08B" for x in pos_broker_nval_last.index],
					["#9467BD" if x in list_max_broker else "#00A08B" for x in pos_broker_nval_last.index],
					["#9467BD" if x in list_max_broker else "#AF0038" for x in neg_broker_nval_last.index],
					["#9467BD" if x in list_max_broker else "#AF0038" for x in neg_broker_nval_last.index],
				],
				line_color="darkslategray",
				align="center",
				font=dict(size=15, color="white"),
				height=30
			)
		),
		row=1, col=2
	)

	# Chart Annotation
	fig.update_layout(title=dict(text=f"<b>{code.upper()}</b>", x=0.5), font=dict(size=20))
	fig.add_annotation(xref="x domain",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1,
		text=f"<b>Broker K Means Clustering | {startdate.strftime('%Y-%m-%d')} - {enddate.strftime('%Y-%m-%d')}</b>",
		font=dict(size=15),align="left",
		showarrow=False
	)
	fig.add_annotation(xref="x domain",yref="paper",xanchor="right",yanchor="bottom",x=1,y=1,
		text=f"<b>Chart by Quantist.io</b>",
		font=dict(size=15),align="right",
		showarrow=False
	)

	# UPDATE_LAYOUT GLOBAL DEFAULT TEMPLATE
	fig.update_layout(legend={"orientation":"h","y":-0.1})
	fig.update_layout(template="plotly_dark",paper_bgcolor="#121212",plot_bgcolor="#121212")

	return fig

async def broker_cluster_timeseries_chart(
	broker_cluster: pd.DataFrame,
	broker_ncum: pd.DataFrame,
	raw_data_close: pd.Series,
	code: str,
	startdate: datetime.date,
	enddate: datetime.date,
	) -> go.Figure:
	# Count number of clusters
	n_clusters = broker_cluster['cluster'].nunique()

	# Sort broker_cluster by correlation
	broker_cluster = broker_cluster.sort_values(by='corr_cluster_abs', ascending=False)
	# Rank correlation (same value will have same rank)
	broker_cluster['rank'] = broker_cluster['corr_cluster_abs'].rank(method='dense', ascending=False)-1

	# Make subplots with max 3 columns with total n_clusters subplots with secondary_y
	n_cols = 3
	n_rows = n_clusters // n_cols
	n_rows = n_rows + 1 if n_clusters % n_cols != 0 else n_rows
	fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True,
		specs=[[{'secondary_y': True}]*n_cols]*n_rows,
		subplot_titles=[f"Cluster {i} (corr:{round(broker_cluster[broker_cluster['rank'] == i]['corr_cluster'][0], 4)})" for i in range(n_clusters)],
		vertical_spacing=0.1, horizontal_spacing=0.1)
	# Add traces to subplots
	for i in range(n_clusters):
		# Get brokers from a cluster
		incluster_brokers = broker_cluster[broker_cluster['rank'] == i].index
		incluster_broker_ncum = broker_ncum[incluster_brokers]
		# Plot line all column of incluster_broker_ncum to subplot
		col_idx = 0
		for col in incluster_broker_ncum.columns:
			fig.add_trace(go.Scatter(
				x=incluster_broker_ncum.index, y=incluster_broker_ncum[col], name=col, line=dict(color='gray'),
				showlegend=False),
				row=i//n_cols+1, col=i%n_cols+1, secondary_y=False)
			# Plot line of mean of incluster_broker_ncum to subplot with yellow color
			fig.add_trace(go.Scatter(
				x=incluster_broker_ncum.index, y=incluster_broker_ncum.mean(axis=1), name='Mean Cluster Transaction', line=dict(color='yellow'),
				showlegend=False if (i != 0 or col_idx != 0) else True),
				row=i//n_cols+1, col=i%n_cols+1, secondary_y=False)
			col_idx += 1
		
		# Plot line of raw_data_close to subplot with red color
		fig.add_trace(go.Scatter(
			x=raw_data_close.index, y=raw_data_close, name='Close Price', line=dict(color='red'),
			showlegend=False if i != 0 else True),
			row=i//n_cols+1, col=i%n_cols+1, secondary_y=True,
			)
		
		# Set subplot title
		fig.update_yaxes(title_text=f"Net Cum Trx", row=i//n_cols+1, col=i%n_cols+1, secondary_y=False)
		fig.update_xaxes(title_text=f"Date", row=i//n_cols+1, col=i%n_cols+1)
		fig.update_yaxes(title_text=f"Price", row=i//n_cols+1, col=i%n_cols+1, secondary_y=True)

	# List the brokers in each cluster from broker_cluster index
	cluster_brokers = [broker_cluster[broker_cluster['rank'] == i].index.tolist() for i in range(n_clusters)]
	# List down the brokers in each cluster below the subplot
	for i in range(n_clusters):
		fig.add_annotation(xref="x domain",yref="y domain",x=0,y=0,
			text=f"<b>{html_wrap(str(cluster_brokers[i]).upper(), width=60, n_lines=4)}</b>",showarrow=False,font=dict(color="white"),
			row=i//n_cols+1, col=i%n_cols+1)

	# Update layout
	fig.update_layout(title=dict(text=f"<b>{code.upper()}</b>", x=0.5, y=0.97, font=dict(size=20)))
	fig.add_annotation(xref="paper",yref="paper",xanchor="left",yanchor="bottom",x=0,y=1.07,
		text = f"<b>Broker Clustering Time Series | {startdate.strftime('%Y-%m-%d')} - {enddate.strftime('%Y-%m-%d')}</b>",
		font=dict(size=15),align="left",
		showarrow=False
	)
	fig.add_annotation(xref="paper",yref="paper",xanchor="right",yanchor="bottom",x=1,y=1.07,
		text=f"<b>Chart by Quantist.io</b>",
		font=dict(size=15),align="right",
		showarrow=False
	)
	
	fig.update_layout(template="plotly_dark",paper_bgcolor="#121212",plot_bgcolor="#121212")
	fig.update_layout(dragmode="pan")
	fig.update_layout(legend={"orientation":"h","y":-0.1})

	return fig

async def fig_to_json(fig:go.Figure):
	return json.dumps(fig, cls=PlotlyJSONEncoder)

async def fig_to_image(fig:go.Figure,format:str = "jpeg"):
	# File Export:
	# fig.write_image("img.jpeg", engine="kaleido", width=1920, height=1080)
	# Bytes Export:
	return fig.to_image(format=format, engine="kaleido", scale=5, width=1280, height=720)