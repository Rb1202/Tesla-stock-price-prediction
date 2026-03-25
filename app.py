import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Tesla AI Predictor", layout="wide")

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_models():
    lstm = load_model("lstm_model.keras")
    try:
        rnn = load_model("rnn_model.keras")
    except:
        rnn = None
    return lstm, rnn

@st.cache_data
def load_scaler():
    return pickle.load(open("scaler.pkl", "rb"))

lstm_model, rnn_model = load_models()
scaler = load_scaler()

# ----------------------------
# Fetch Live Tesla Data
# ----------------------------
@st.cache_data
def load_live_data():
    df = yf.download("TSLA", period="1y", interval="1d")
    return df

df_live = load_live_data()

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<h1 style='text-align:center;color:#4CAF50;'>🚗 Tesla AI Stock Predictor</h1>
<p style='text-align:center;'>Live Data + Deep Learning (RNN vs LSTM)</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox("Model", ["LSTM", "SimpleRNN", "Compare Both"])
days = st.sidebar.slider("Prediction Days", 1, 10, 5)

# ----------------------------
# Prepare Data
# ----------------------------
close_data = df_live[['Close']]   # keep as DataFrame
scaled_data = scaler.transform(close_data)
last_60_days = scaled_data[-60:]

from sklearn.metrics import mean_squared_error

def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(scaled_data)

# Train-test split
split = int(0.8 * len(X))
X_test = X[split:]
y_test = y[split:]

# Predictions
lstm_pred_test = lstm_model.predict(X_test, verbose=0)
y_test_actual = scaler.inverse_transform(y_test)
lstm_pred_test = scaler.inverse_transform(lstm_pred_test)

if rnn_model:
    rnn_pred_test = rnn_model.predict(X_test, verbose=0)
    rnn_pred_test = scaler.inverse_transform(rnn_pred_test)

# MSE Calculation
lstm_mse = mean_squared_error(y_test_actual, lstm_pred_test)

if rnn_model:
    rnn_mse = mean_squared_error(y_test_actual, rnn_pred_test)
else:
    rnn_mse = None
# ----------------------------
# Prediction Function
# ----------------------------
def predict_days(model, data, days):
    temp = data.copy()
    preds = []

    for _ in range(days):
        pred = model.predict(temp.reshape(1,60,1), verbose=0)
        preds.append(pred[0][0])
        temp = np.vstack((temp[1:], pred))

    return scaler.inverse_transform(np.array(preds).reshape(-1,1))

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns([2,1])

# ----------------------------
# Candlestick Chart
# ----------------------------
with col1:
    st.subheader("📊 Live Tesla Stock (Candlestick)")

    st.line_chart(df_live['Close'])  # lightweight candlestick alternative

# ----------------------------
# Predict
# ----------------------------
if st.button("🚀 Predict Now"):

    with col1:
        st.subheader("📈 Prediction Graph")

    with col2:
        st.subheader("📌 Summary")

    # ----------------------------
    # CASE 1: Compare Both
    # ----------------------------
    if model_choice == "Compare Both" and rnn_model is not None:

        lstm_pred = predict_days(lstm_model, last_60_days, days)
        rnn_pred = predict_days(rnn_model, last_60_days, days)

        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(lstm_pred, label="LSTM")
            ax.plot(rnn_pred, label="RNN")
            ax.legend()
            ax.set_title("Model Comparison")
            st.pyplot(fig, use_container_width=False)

        with col2:
            st.write("### LSTM Predictions")
            for i, val in enumerate(lstm_pred):
                st.metric(f"LSTM Day {i+1}", f"{val[0]:.2f}")

            st.write("### RNN Predictions")
            for i, val in enumerate(rnn_pred):
                st.metric(f"RNN Day {i+1}", f"{val[0]:.2f}")

    # ----------------------------
    # CASE 2: Single Model
    # ----------------------------
    else:

        if model_choice == "LSTM":
            model = lstm_model
        elif model_choice == "SimpleRNN" and rnn_model is not None:
            model = rnn_model
        else:
            st.warning("RNN not available → using LSTM")
            model = lstm_model

        preds = predict_days(model, last_60_days, days)

        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(range(1, days+1), preds.flatten(), marker='o')
            ax.set_title("Future Price Trend")
            ax.set_xlabel("Days")
            ax.set_ylabel("Price")
            st.pyplot(fig, use_container_width=False)

        with col2:
            for i, val in enumerate(preds):
                st.metric(f"Day {i+1}", f"{val[0]:.2f}")

st.subheader("📊 Model Accuracy Comparison")

st.write(f"LSTM MSE: {lstm_mse:.4f}")

if rnn_mse is not None:
    st.write(f"RNN MSE: {rnn_mse:.4f}")

    if lstm_mse < rnn_mse:
        st.success("✅ LSTM is more accurate")
    else:
        st.success("✅ RNN is more accurate")
else:
    st.info("RNN model not available")
# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("<center>⚡ Powered by Streamlit + Deep Learning</center>", unsafe_allow_html=True)