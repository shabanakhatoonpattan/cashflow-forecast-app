# cashflow_app.py

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------
# Streamlit App Title
st.title("💰 Cash Flow Forecast & Monte Carlo Simulation")

# ---------------------------
# 1️⃣ Upload Excel
uploaded_file = st.file_uploader("Upload your CashFlow Excel", type=["xlsx"])
if uploaded_file is not None:
    cashflow_df = pd.read_excel(uploaded_file, sheet_name='CashFlow')
    st.subheader("Preview of uploaded data")
    st.dataframe(cashflow_df.head())

    # Ensure Month is datetime
    if 'Month' in cashflow_df.columns:
        try:
            cashflow_df['Month'] = pd.to_datetime(cashflow_df['Month'], errors='coerce')
        except Exception as e:
            st.error(f"Failed to parse Month column: {e}")
    else:
        st.error("No 'Month' column found in Excel!")

    # Ensure Revenue column exists and has numeric values
    if 'Revenue (GBP)' in cashflow_df.columns:
        cashflow_df['Revenue (GBP)'] = pd.to_numeric(cashflow_df['Revenue (GBP)'], errors='coerce').fillna(0)
    else:
        st.error("No 'Revenue (GBP)' column found in Excel!")

    # ---------------------------
    # 2️⃣ ARIMA Forecast
    st.subheader("📈 Revenue Forecast")
    revenue_series = cashflow_df['Revenue (GBP)']
    if revenue_series.sum() == 0:
        st.warning("Revenue series is all zeros. Forecast may not be meaningful.")
    else:
        try:
            arima_model = ARIMA(revenue_series, order=(1,1,1))
            model_fit = arima_model.fit()
            forecast_steps = st.slider("Months to forecast", 1, 12, 6)
            forecast_values = model_fit.forecast(steps=forecast_steps)
            
            forecast_months = pd.date_range(
                start=cashflow_df['Month'].iloc[-1] + pd.offsets.MonthBegin(),
                periods=forecast_steps, freq='MS'
            )
            forecast_df = pd.DataFrame({'Month': forecast_months, 'Forecasted Revenue': forecast_values})
            
            # Combine historical + forecast for plotting
            plot_df = pd.concat([
                cashflow_df.set_index('Month')['Revenue (GBP)'], 
                forecast_df.set_index('Month')['Forecasted Revenue']
            ])
            st.line_chart(plot_df)
            
            # Export forecast
            st.download_button(
                label="📥 Download Forecast as Excel",
                data=forecast_df.to_excel(index=False, engine='openpyxl'),
                file_name='forecasted_revenue.xlsx'
            )
        except Exception as e:
            st.error(f"Forecasting failed: {e}")

    # ---------------------------
    # 3️⃣ Monte Carlo Simulation
    st.subheader("🎲 Monte Carlo Simulation for Closing Cash")
    n_sim = st.slider("Number of simulations", 100, 5000, 1000)
    base_hourly_rate = st.number_input("Base hourly rate (£)", value=100)
    base_hours = st.number_input("Base hours per month", value=120)
    opening_cash = st.number_input("Opening cash (£)", value=5000)

    # Get fixed expenses if columns exist
    expense_cols = ['Software (GBP)','Marketing (GBP)','Rent/Overheads (GBP)']
    fixed_expenses = cashflow_df[expense_cols].iloc[0].sum() if all(col in cashflow_df.columns for col in expense_cols) else 0

    simulated_cash_flows = []
    for _ in range(n_sim):
        hourly_rate = base_hourly_rate * np.random.uniform(0.95, 1.05)
        hours = base_hours * np.random.uniform(0.9, 1.1)
        revenue = hourly_rate * hours
        total_expense = revenue * 0.10 + fixed_expenses
        profit = revenue - total_expense
        net_cash = opening_cash + profit
        simulated_cash_flows.append(net_cash)
    
    st.bar_chart(pd.Series(simulated_cash_flows))
