import streamlit as st
import pandas as pd
import numpy as np
import itertools
from plotly import graph_objs as go
from datetime import timedelta
from aemet import Aemet
from config import API_KEY
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn import metrics

stations = {
    #'8058X': 'OLIVA',
    #'8325X': 'POLINYÀ DE XÚQUER',
    "8309X": "UTIEL",
    "8414A": "VALENCIA AEROPUERTO",
    #'8416Y': 'VALÈNCIA, VIVEROS',
    #'8416': 'VALÈNCIA',
    "8293X": "XÀTIVA",
}
cutoffs = pd.to_datetime(["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31"])


def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha"], y=df["tmed"], name="tmed"))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperatura media",
        title="Time Series data",
        height=600,
        width=1000,
    )
    st.plotly_chart(fig)


def plot_forecast(n_days, forecast, y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=forecast["ds"].tail(n_days), y=y_test, mode="lines", name="Real")
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"].tail(n_days),
            y=y_pred,
            mode="lines",
            name="Forecast",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"].tail(n_days),
            y=forecast["yhat_upper"].tail(n_days),
            mode="lines",
            line_color = 'rgba(0,0,0,0)',
            fillcolor="rgba(255,153,51,0.2)", showlegend = False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"].tail(n_days),
            y=forecast["yhat_lower"].tail(n_days),
            mode="lines",
            line_color = 'rgba(0,0,0,0)',
            fill="tonexty",
            fillcolor="rgba(255,153,51,0.2)",
            name="Confidence Bound"
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperatura media",
        title="Facebook Prophet forecast",
        height=600,
        width=1000,
    )
    st.plotly_chart(fig)


def load_data(station_code, fechaini, fechafin):
    aemet = Aemet(api_key=API_KEY)
    data = aemet.get_valores_climatologicos_diarios(
        fechaini=fechaini.strftime("%Y-%m-%dT%H:%M:%SUTC"),
        fechafin=fechafin.strftime("%Y-%m-%dT%H:%M:%SUTC"),
        estacion=station_code,
    )
    df = pd.DataFrame(data)
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d")
    numeric_columns = [
        "tmed",
        "tmin",
        "tmax",
        "velmedia",
        "racha",
        "sol",
        "presMax",
        "presMin",
    ]
    for column in numeric_columns:
        df[column] = df[column].str.replace(",", ".").astype(float)
    return df


def load_new_data(station_code, fechafin, n_days):
    aemet = Aemet(api_key=API_KEY)
    data = aemet.get_valores_climatologicos_diarios(
        fechaini=(fechafin + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SUTC"),
        fechafin=(fechafin + timedelta(days=n_days)).strftime("%Y-%m-%dT%H:%M:%SUTC"),
        estacion=station_code,
    )
    df_new = pd.DataFrame(data)
    df_new["fecha"] = pd.to_datetime(df_new["fecha"], format="%Y-%m-%d")
    numeric_columns = [
        "tmed",
        "tmin",
        "tmax",
        "velmedia",
        "racha",
        "sol",
        "presMax",
        "presMin",
    ]
    for column in numeric_columns:
        df_new[column] = df_new[column].str.replace(",", ".").astype(float)
    return df_new


def hyperparameter_tuning(df_train):
    param_grid = {
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
        "holidays_prior_scale": [0.01, 10],
        "seasonality_mode": ["additive", "multiplicative"],
    }

    all_params = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]
    rmses = []
    progress_bar = st.progress(0)

    for i, params in enumerate(all_params):
        model = Prophet(**params).fit(df_train)
        df_cv = cross_validation(model, cutoffs=cutoffs, horizon="200 days")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p["rmse"].values[0])
        progress = (i + 1) / len(all_params)
        progress_bar.progress(progress)

    tuning_results = pd.DataFrame(all_params)
    tuning_results["rmse"] = rmses
    best_params = all_params[np.argmin(rmses)]
    return best_params


def calculate_metrics(y_pred, y_test):
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred) * 100
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    data = {
        "MAE": [round(mae, 4)],
        "MAPE (%)": [round(mape, 4)],
        "MSE": [round(mse, 4)],
        "RMSE": [round(rmse, 4)],
    }
    df_errors = pd.DataFrame(data)
    return df_errors
