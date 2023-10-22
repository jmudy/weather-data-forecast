import streamlit as st
from utils import *
from prophet import Prophet
from datetime import date
from prophet.plot import plot_plotly
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

st.markdown(
	"""
	<style>
		section.main > div {max-width:70rem}
	</style>
	""", unsafe_allow_html=True
	)

st.sidebar.title('Settings')
st.sidebar.markdown('---')

with st.sidebar.expander('**Station**'):
	station = st.selectbox('**Select station for prediction**', stations, index=None)

with st.sidebar.expander('**Date**'):
	fechaini = st.date_input('**Select a start date**', date(2018,1,1))
	fechafin = st.date_input('**Select an end date**', date(2022,12,31))

n_days = st.sidebar.slider('**Days of prediction:**',min_value=1, max_value=365, value=200)
tuning = st.sidebar.checkbox('Hyperparameter tuning')
forecast = st.sidebar.button('Make forecast', use_container_width=True)

# Aquí meter un selector de hiperparámetros para seleccionarlos manualmente si no se elige 'tuning'

st.title('Weather data forecast in Valencia')
st.markdown('---')

if station:

	if forecast:

		if tuning:

			with st.spinner(text='Loading data...'):
				df = load_data(station, fechaini, fechafin)
			st.success('El dataframe df ha sido cargado satisfactoriamente!')

			st.subheader('Raw data')
			st.markdown('---')
			st.dataframe(df.tail(10), use_container_width=True)

			df_train = df[['fecha','tmed']]
			df_train = df_train.rename(columns={'fecha': 'ds', 'tmed': 'y'})

			plot_raw_data(df)

			# Carga de los datos de test
			df_test = load_new_data(station, fechafin, n_days)

			with st.spinner(text='Calculando hiperparámetros óptimos...'):
				best_params = hyperparameter_tuning(df_train)
			st.success('Hiperparámetros óptimos calculados satisfactoriamente!')

			model = Prophet(**best_params)
			model.fit(df_train)
			future = model.make_future_dataframe(periods=n_days)
			forecast = model.predict(future)

			st.subheader('Forecast data')
			st.markdown('---')
			st.dataframe(forecast[['ds', 'trend_lower', 'trend', 'trend_upper', 'yhat_lower', 'yhat', 'yhat_upper']].tail(n_days), use_container_width=True)

			st.write(f'Forecast plot for {n_days} days')
			fig1 = plot_plotly(model, forecast)
			#fig1.update_traces(line=dict(color='rgba(110, 135, 255, 0.5)')) # color de la traza
			fig1.update_traces(marker=dict(color='rgb(110, 135, 255)'), selector=dict(mode='markers')) # color puntos observaciones reales
			fig1.update_traces(fillcolor='rgba(110, 135, 255, 0.3)') # color bandas de confianza
			st.plotly_chart(fig1)

			st.subheader('Metrics')
			st.markdown('---')
			y_pred = forecast['yhat'].tail(n_days).reset_index(drop=True)
			y_test = df_test['tmed'].head(n_days)

			df_errors = calculate_metrics(y_pred, y_test)
			st.dataframe(df_errors, use_container_width=True)

			plot_forecast(n_days, forecast, y_test, y_pred)

		else:

			with st.spinner(text='Loading data...'):
				df = load_data(station, fechaini, fechafin)
			st.success('El dataframe df ha sido cargado satisfactoriamente!')

			st.subheader('Raw data')
			st.markdown('---')
			st.dataframe(df.tail(10), use_container_width=True)

			df_train = df[['fecha','tmed']]
			df_train = df_train.rename(columns={'fecha': 'ds', 'tmed': 'y'})

			plot_raw_data(df)

			# Carga de los datos de test
			df_new = load_new_data(station, fechafin, n_days)

			model = Prophet()
			model.fit(df_train)
			future = model.make_future_dataframe(periods=n_days)
			forecast = model.predict(future)

			st.subheader('Forecast data')
			st.markdown('---')
			st.dataframe(forecast[['ds', 'trend_lower', 'trend', 'trend_upper', 'yhat_lower', 'yhat', 'yhat_upper']].tail(n_days), use_container_width=True)

			st.write(f'Forecast plot for {n_days} days')
			fig1 = plot_plotly(model, forecast)
			#fig1.update_traces(line=dict(color='rgba(110, 135, 255, 0.5)')) # color de la traza
			fig1.update_traces(marker=dict(color='rgb(110, 135, 255)'), selector=dict(mode='markers')) # color puntos observaciones reales
			fig1.update_traces(fillcolor='rgba(110, 135, 255, 0.3)') # color bandas de confianza
			st.plotly_chart(fig1)

			st.subheader('Metrics')
			st.markdown('---')
			y_pred = forecast['yhat'].tail(n_days).reset_index(drop=True)
			y_test = df_new['tmed'].head(n_days)

			df_errors = calculate_metrics(y_pred, y_test)
			st.dataframe(df_errors, use_container_width=True)

			plot_forecast(n_days, forecast, y_test, y_pred)

	else:
		st.info('Select if you want to adjust the hyperparameters automatically.')

else:
	st.info('Please, select a station.')
