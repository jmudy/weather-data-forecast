import streamlit as st

from aemet import Aemet
import pandas as pd
from config import API_KEY, FECHAINI, FECHAFIN

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
	data = aemet.get_valores_climatologicos_diarios(fechaini=FECHAINI, fechafin=FECHAFIN, estacion=station)
	df = pd.DataFrame(data)
	df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
	numeric_columns = ['tmed','tmin','tmax','velmedia','racha','sol','presMax','presMin']
	for column in numeric_columns:
		df[column] = df[column].str.replace(',', '.').astype(float)
	return df

if forecast:

	data_load_state = st.text('Loading data...')
	df = load_data()
	data_load_state.text('Loading data... done!')

	df_train = df[['fecha','tmed']]
	df_train = df_train.rename(columns={"fecha": "ds", "tmed": "y"})

	st.subheader('Raw data')
	st.write(df.tail(10))

	plot_raw_data()

	model = Prophet()
	model.fit(df_train)
	future = model.make_future_dataframe(periods=n_days)
	forecast = model.predict(future)

	st.subheader('Forecast data')
	st.write(forecast[['ds', 'trend_lower', 'trend', 'trend_upper', 'yhat_lower', 'yhat', 'yhat_upper']].tail(10))
		
	st.write(f'Forecast plot for {n_days} days')
	fig1 = plot_plotly(model, forecast)
	st.plotly_chart(fig1)

	# Sustituir esto por una tabla resumen con las métricas más relevantes
	st.subheader("Forecast components")
	fig2 = model.plot_components(forecast)
	st.write(fig2)