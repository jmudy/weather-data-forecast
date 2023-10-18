import streamlit as st

from aemet import Aemet
import pandas as pd
from datetime import date, timedelta
from config import API_KEY

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

st.markdown(
	"""
	<style>
		section.main > div {max-width:70rem}
	</style>
	""", unsafe_allow_html=True
	)

st.title('Weather data forecast in Valencia')

stations = ('8058X', '8325X', '8309X', '8414A', '8416Y', '8416', '8293X')

st.sidebar.title('Settings')
st.sidebar.markdown('---')
station = st.sidebar.selectbox('**Select station for prediction**', stations, index=None)
st.sidebar.markdown('---')
fechaini = st.sidebar.date_input('**Select a start date**', date(2018,1,1))
fechafin = st.sidebar.date_input('**Select an end date**', date(2022,12,31))
st.sidebar.markdown('---')
n_days = st.sidebar.slider('**Days of prediction:**',min_value=1, max_value=60, value=40)

if station:
	forecast = st.sidebar.button('Make forecast')

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['fecha'], y=df['tmed'], name="tmed"))
	fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

@st.cache_data
def load_data():
	aemet = Aemet(api_key=API_KEY)
	data = aemet.get_valores_climatologicos_diarios(fechaini=fechaini.strftime("%Y-%m-%dT%H:%M:%SUTC"),
												    fechafin=fechafin.strftime("%Y-%m-%dT%H:%M:%SUTC"),
													estacion=station)
	df = pd.DataFrame(data)
	df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
	numeric_columns = ['tmed','tmin','tmax','velmedia','racha','sol','presMax','presMin']
	for column in numeric_columns:
		df[column] = df[column].str.replace(',', '.').astype(float)
	return df

@st.cache_data
def load_new_data():
	aemet = Aemet(api_key=API_KEY)
	data = aemet.get_valores_climatologicos_diarios(fechaini=(fechafin + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SUTC"),
												    fechafin=(fechafin + timedelta(days=n_days)).strftime("%Y-%m-%dT%H:%M:%SUTC"),
												    estacion=station)
	df_new = pd.DataFrame(data)
	df_new['fecha'] = pd.to_datetime(df_new['fecha'], format='%Y-%m-%d')
	numeric_columns = ['tmed','tmin','tmax','velmedia','racha','sol','presMax','presMin']
	for column in numeric_columns:
		df_new[column] = df_new[column].str.replace(',', '.').astype(float)
	return df_new

if forecast:

	data_load_state = st.text('Loading data...')
	df = load_data()
	data_load_state.text('Loading data... done!')

	df_train = df[['fecha','tmed']]
	df_train = df_train.rename(columns={"fecha": "ds", "tmed": "y"})

	st.subheader('Raw data')
	st.dataframe(df.tail(10))

	st.subheader('Raw new data')
	df_new = load_new_data()
	st.dataframe(df_new)

	plot_raw_data()

	model = Prophet()
	model.fit(df_train)
	future = model.make_future_dataframe(periods=n_days)
	forecast = model.predict(future)

	st.subheader('Forecast data')
	st.dataframe(forecast[['ds', 'trend_lower', 'trend', 'trend_upper', 'yhat_lower', 'yhat', 'yhat_upper']].tail(n_days))
		
	st.write(f'Forecast plot for {n_days} days')
	fig1 = plot_plotly(model, forecast)
	st.plotly_chart(fig1)

	# Sustituir esto por una tabla resumen con las métricas más relevantes
	st.subheader("Forecast components")
	fig2 = model.plot_components(forecast)
	st.write(fig2)
