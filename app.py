import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_page_config(
    page_title="Stock Price Forecast",
    layout="wide"
)

st.title("📈 Stock Price Forecast using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if st.button("Analyze"):

    with st.spinner("Downloading data..."):

        data = yf.download(
            ticker,
            start="2021-01-01",
            end="2026-06-24",
            auto_adjust=True
        )

    if data.empty:
        st.error("No data found.")
        st.stop()

    prices = data["Close"]

    st.subheader("Historical Price Data")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        prices.index,
        prices.values,
        linewidth=1.5
    )

    ax.set_title(
        f"{ticker} Closing Price (Last 5 Years)"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("ARIMA Model Training")

    train_size = int(len(prices) * 0.8)

    train = prices[:train_size]
    test = prices[train_size:]

    model = ARIMA(
        train,
        order=(5, 1, 0)
    )

    model_fit = model.fit()

    predictions = model_fit.forecast(
        steps=len(test)
    )

    rmse = sqrt(
        mean_squared_error(
            test,
            predictions
        )
    )

    st.success(
        f"Model RMSE: {rmse:.2f}"
    )

    st.subheader("Future Forecast")

    final_model = ARIMA(
        prices,
        order=(5, 1, 0)
    )

    final_fit = final_model.fit()

    forecast_days = 365

    forecast = final_fit.forecast(
        steps=forecast_days
    )

    forecast_dates = pd.date_range(
        start=prices.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast_df = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Forecast": forecast
        }
    )

    june_2027 = forecast_df[
        (forecast_df["Date"].dt.year == 2027)
        &
        (forecast_df["Date"].dt.month == 6)
    ]

    june_price = float(
        june_2027.iloc[0]["Forecast"]
    )

    st.metric(
        "Predicted Price (June 2027)",
        f"${june_price:.2f}"
    )

    fig2, ax2 = plt.subplots(
        figsize=(14, 6)
    )

    ax2.plot(
        prices.index,
        prices,
        label="Historical"
    )

    ax2.plot(
        forecast_df["Date"],
        forecast_df["Forecast"],
        label="Forecast"
    )

    ax2.legend()

    ax2.set_title(
        f"{ticker} Forecast till June 2027"
    )

    ax2.grid(True)

    st.pyplot(fig2)

    st.subheader("Forecast Data")

    st.dataframe(
        forecast_df.tail(30)
    )
