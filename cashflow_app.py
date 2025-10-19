# cashflow_app_interactive.py

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import altair as alt
import io

# ---------------------------
# Page config
st.set_page_config(
    page_title="Cash Flow Forecast",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Cash Flow Forecast & Monte Carlo Simulation by Shabana Pattan")

# ---------------------------
# Sidebar controls
st.sidebar.header("Simulation & Forecast Parameters")
n_sim = st.sidebar.slider("Number of Monte Carlo simulations", 100, 5000, 1000)
base_hourly_rate = st.sidebar.number_input("Base hourly rate (£)", value=100)
base_hours = st.sidebar.number_input("Base hours per month", value=120)
opening_cash = st.sidebar.number_input("Opening cash (£)", value=5000)
forecast_steps = st.sidebar.slider("Months to forecast", 1, 12, 6)

# ---------------------------
# 1️⃣ Upload Excel
uploaded_file = st.file_uploader("Upload your CashFlow Excel", type=["xlsx"])
if uploaded_file is not None:
    cashflow_df = pd.read_excel(uploaded_file, sheet_name='CashFlow')
    st.subheader("Preview of uploaded data")
    st.dataframe(cashflow_df.head())

    # Validate Month column
    if 'Month' in cashflow_df.columns:
        try:
            cashflow_df['Month'] = pd.to_datetime(cashflow_df['Month'], errors='coerce')
            if cashflow_df['Month'].isna().any():
                st.warning("Some months could not be parsed. Check Excel format.")
        except Exception as e:
            st.error(f"Failed to parse Month column: {e}")
    else:
        st.error("No 'Month' column found in Excel!")

    # Validate Revenue column
    if 'Revenue (GBP)' in cashflow_df.columns:
        cashflow_df['Revenue (GBP)'] = pd.to_numeric(cashflow_df['Revenue (GBP)'], errors='coerce').fillna(0)
    else:
        st.error("No 'Revenue (GBP)' column found in Excel!")

    # ---------------------------
    # 2️⃣ ARIMA Forecast
    st.subheader("📈 Revenue Forecast")
    revenue_series = cashflow_df['Revenue (GBP)']
    if revenue_series.sum() == 0:
        st.warning("Revenue data is all zeros. Forecast may not be meaningful.")
    else:
        try:
            arima_model = ARIMA(revenue_series, order=(1,1,1))
            model_fit = arima_model.fit()
            forecast_values = model_fit.forecast(steps=forecast_steps)

            forecast_months = pd.date_range(
                start=cashflow_df['Month'].iloc[-1] + pd.offsets.MonthBegin(),
                periods=forecast_steps, freq='MS'
            )
            forecast_df = pd.DataFrame({'Month': forecast_months, 'Forecasted Revenue': forecast_values})

            # Combine historical + forecast
            plot_df = pd.concat([
                cashflow_df.set_index('Month')['Revenue (GBP)'], 
                forecast_df.set_index('Month')['Forecasted Revenue']
            ]).reset_index().melt(id_vars='Month', var_name='Type', value_name='Revenue')

            chart = alt.Chart(plot_df).mark_line(point=True).encode(
                x='Month:T',
                y='Revenue:Q',
                color='Type:N',
                tooltip=['Month:T','Type:N','Revenue:Q']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

            # Download forecast as Excel
            output = io.BytesIO()
            forecast_df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            st.download_button(
                label="📥 Download Forecast as Excel",
                data=output,
                file_name="forecasted_revenue.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Forecasting failed: {e}")

    # ---------------------------
    # 3️⃣ Monte Carlo Simulation
    st.subheader("🎲 Monte Carlo Simulation for Closing Cash")

    # Fixed expenses
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

    sim_series = pd.Series(simulated_cash_flows)

    st.write("**Monte Carlo Summary:**")
    st.write(f"- Mean closing cash: £{sim_series.mean():,.2f}")
    st.write(f"- Median closing cash: £{sim_series.median():,.2f}")
    st.write(f"- Min closing cash: £{sim_series.min():,.2f}")
    st.write(f"- Max closing cash: £{sim_series.max():,.2f}")

    # Histogram with Altair
    hist_chart = alt.Chart(sim_series.reset_index()).mark_bar().encode(
        alt.X("0:Q", bin=alt.Bin(maxbins=30), title="Closing Cash (£)"),
        y='count()'
    )
    st.altair_chart(hist_chart, use_container_width=True)
