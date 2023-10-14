import streamlit as st

from aemet import Aemet, Estacion
import pandas as pd
from config import API_KEY, FECHAINI, FECHAFIN

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.markdown("""
<style>
    section.main > div {max-width:70rem}
</style>
""", unsafe_allow_html=True)

st.title('Weather data forecast in Valencia')

stations = ('8058X', '8325X', '8309X', '8414A', '8416Y', '8416', '8293X')

st.sidebar.title('Settings')
st.sidebar.markdown('---')
station = st.sidebar.selectbox('**Select station for prediction**', stations, index=None)
st.sidebar.markdown('---')
n_years = st.sidebar.slider('**Years of prediction:**',min_value=1, max_value=4, value=2)
period = n_years * 365

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['fecha'], y=df['tmed'], name="tmed"))
	fig.add_trace(go.Scatter(x=df['fecha'], y=df['tmax'], name="tmax"))
	fig.add_trace(go.Scatter(x=df['fecha'], y=df['tmin'], name="tmin"))
	fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

if station:

	data_load_state = st.text('Loading data...')

	aemet = Aemet(api_key=API_KEY)
	estaciones = Estacion.get_estaciones(api_key=API_KEY)

	data = aemet.get_valores_climatologicos_diarios(fechaini=FECHAINI, fechafin=FECHAFIN, estacion=station)
	df = pd.DataFrame(data)
	numeric_columns = ['tmed','prec','tmin','tmax','velmedia','racha','sol','presMax','presMin']
	for column in numeric_columns:
		df[column] = df[column].str.replace(',', '.')	

	data_load_state.text('Loading data... done!')

	st.subheader('Raw data')
	st.write(df.tail(10))
		
	plot_raw_data()

	df_train = df[['fecha','tmed']]
	df_train = df_train.rename(columns={"fecha": "ds", "tmed": "y"})

	m = Prophet()
	m.fit(df_train)
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)

	st.subheader('Forecast data')
	st.write(forecast.tail(10))
		
	st.write(f'Forecast plot for {n_years} years')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)

	st.subheader("Forecast components")
	fig2 = m.plot_components(forecast)
	st.write(fig2)