import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(page_title="Investment Tax Model Comparison", layout="wide")

st.title("🏦 Investment Tax Model Comparison")
st.markdown("Compare three different taxation models on investment returns using MSCI World Index data")

# Sidebar for parameters
st.sidebar.header("📊 Model Parameters")

st.sidebar.subheader("Investment Settings")
initial_investment = st.sidebar.number_input("Initial Investment (EUR)", value=100000, step=1000, min_value=0)
monthly_contribution = st.sidebar.number_input("Monthly Contribution (EUR)", value=1000, step=100, min_value=0)

st.sidebar.subheader("Forfait Model")
forfait_r = st.sidebar.slider("Deemed Return Rate (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
forfait_tax_rate = st.sidebar.slider("Forfait Tax Rate (%)", min_value=0.0, max_value=100.0, value=36.0, step=1.0)

st.sidebar.subheader("Accrual Model")
accrual_tax_rate = st.sidebar.slider("Accrual Tax Rate (%)", min_value=0.0, max_value=100.0, value=36.0, step=1.0)

st.sidebar.subheader("Labor Model")
labor_tax_rate = st.sidebar.slider("Labor Tax Rate (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
wealth_tax_rate = st.sidebar.slider("Wealth Tax Rate (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
wealth_tax_threshold = st.sidebar.number_input("Wealth Tax Threshold (EUR)", value=200000, step=10000, min_value=0)

# Function to fetch MSCI World data
@st.cache_data
def fetch_msci_data(start_date, end_date):
    """Fetch MSCI World Index data. Try multiple tickers."""
    tickers_to_try = ['URTH', 'ACWI', 'IWDA.AS']  # iShares MSCI World ETFs
    
    for ticker in tickers_to_try:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                st.sidebar.success(f"✓ Using {ticker} as MSCI World proxy")
                return data['Adj Close']
        except Exception as e:
            st.sidebar.warning(f"Failed to fetch {ticker}: {str(e)}")
            continue
    
    st.error("Could not fetch MSCI World data from any source. Please check your internet connection.")
    return None

# Fetch data
start_date = "2021-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")  # Use today's date

with st.spinner("Fetching MSCI World Index data..."):
    price_data = fetch_msci_data(start_date, end_date)

if price_data is not None:
    # Prepare daily data
    df = pd.DataFrame(price_data)
    df.columns = ['price']
    df = df.reset_index()
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to exact date range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if df.empty:
        st.error("No data available for the specified date range")
        st.stop()
    
    # Calculate daily returns
    df['daily_return'] = df['price'].pct_change()
    
    # Create month-end flags
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['is_month_end'] = df.groupby(['year', 'month'])['date'].transform('max') == df['date']
    df['is_year_start'] = (df['date'].dt.month == 1) & (df['date'].dt.day == 1)
    df['is_year_end'] = (df['date'].dt.month == 12) & (df['date'].dt.day == 31)
    
    # If Dec 31 doesn't exist, use last trading day of December
    for year in df['year'].unique():
        if not any((df['year'] == year) & (df['month'] == 12) & df['is_year_end']):
            dec_data = df[(df['year'] == year) & (df['month'] == 12)]
            if not dec_data.empty:
                last_dec_idx = dec_data['date'].idxmax()
                df.loc[last_dec_idx, 'is_year_end'] = True
    
    # Initialize tracking dataframes
    def simulate_forfait(df, initial, monthly_contrib, r_rate, tax_rate):
        """Simulate Forfait model"""
        portfolio_value = initial
        shares = initial / df.iloc[0]['price']
        cumulative_tax = 0
        
        results = []
        
        for idx, row in df.iterrows():
            # Apply daily return to shares
            if idx > 0:
                shares = shares * (1 + row['daily_return'])
            
            portfolio_value = shares * row['price']
            
            # Annual tax on Jan 1 (calculated on previous year-end value)
            if row['is_year_start'] and row['year'] > 2021:
                tax_base = portfolio_value
                tax_amount = tax_base * (r_rate / 100) * (tax_rate / 100)
                cumulative_tax += tax_amount
                # Pay tax by reducing shares
                shares_to_sell = tax_amount / row['price']
                shares -= shares_to_sell
                portfolio_value = shares * row['price']
            
            # Monthly contribution at month end
            if row['is_month_end'] and not (row['year'] == 2021 and row['month'] == 1):
                new_shares = monthly_contrib / row['price']
                shares += new_shares
                portfolio_value = shares * row['price']
            
            results.append({
                'date': row['date'],
                'portfolio_value': portfolio_value,
                'cumulative_tax': cumulative_tax,
                'shares': shares
            })
        
        return pd.DataFrame(results)
    
    def simulate_accrual(df, initial, monthly_contrib, tax_rate):
        """Simulate Accrual model"""
        portfolio_value = initial
        shares = initial / df.iloc[0]['price']
        cumulative_tax = 0
        loss_carryforward = 0
        year_start_value = initial
        
        results = []
        
        for idx, row in df.iterrows():
            # Apply daily return
            if idx > 0:
                shares = shares * (1 + row['daily_return'])
            
            portfolio_value = shares * row['price']
            
            # Annual tax on Dec 31
            if row['is_year_end']:
                gain = portfolio_value - year_start_value
                
                if gain > 0:
                    # Apply loss carryforward first
                    taxable_gain = max(0, gain - loss_carryforward)
                    loss_carryforward = max(0, loss_carryforward - gain)
                    
                    if taxable_gain > 0:
                        tax_amount = taxable_gain * (tax_rate / 100)
                        cumulative_tax += tax_amount
                        # Pay tax by reducing shares
                        shares_to_sell = tax_amount / row['price']
                        shares -= shares_to_sell
                        portfolio_value = shares * row['price']
                else:
                    # Loss - add to carryforward
                    loss_carryforward += abs(gain)
            
            # Set year start value for next year
            if row['is_year_end']:
                year_start_value = portfolio_value
            
            # Monthly contribution at month end
            if row['is_month_end'] and not (row['year'] == 2021 and row['month'] == 1):
                new_shares = monthly_contrib / row['price']
                shares += new_shares
                portfolio_value = shares * row['price']
                # Adjust year start value for contributions
                if row['is_year_end']:
                    year_start_value = portfolio_value
            
            results.append({
                'date': row['date'],
                'portfolio_value': portfolio_value,
                'cumulative_tax': cumulative_tax,
                'shares': shares,
                'loss_carryforward': loss_carryforward
            })
        
        return pd.DataFrame(results)
    
    def simulate_labor(df, initial, monthly_contrib, tax_rate, wealth_tax_rate, wealth_threshold):
        """Simulate Labor model (Accrual + Wealth Tax)"""
        portfolio_value = initial
        shares = initial / df.iloc[0]['price']
        cumulative_tax = 0
        cumulative_income_tax = 0
        cumulative_wealth_tax = 0
        loss_carryforward = 0
        year_start_value = initial
        
        results = []
        
        for idx, row in df.iterrows():
            # Apply daily return
            if idx > 0:
                shares = shares * (1 + row['daily_return'])
            
            portfolio_value = shares * row['price']
            
            # Annual taxes on Dec 31
            if row['is_year_end']:
                # Income tax on gains (accrual basis)
                gain = portfolio_value - year_start_value
                
                if gain > 0:
                    taxable_gain = max(0, gain - loss_carryforward)
                    loss_carryforward = max(0, loss_carryforward - gain)
                    
                    if taxable_gain > 0:
                        income_tax = taxable_gain * (tax_rate / 100)
                        cumulative_tax += income_tax
                        cumulative_income_tax += income_tax
                        shares_to_sell = income_tax / row['price']
                        shares -= shares_to_sell
                        portfolio_value = shares * row['price']
                else:
                    loss_carryforward += abs(gain)
                
                # Wealth tax
                if portfolio_value > wealth_threshold:
                    wealth_tax = (portfolio_value - wealth_threshold) * (wealth_tax_rate / 100)
                    cumulative_tax += wealth_tax
                    cumulative_wealth_tax += wealth_tax
                    shares_to_sell = wealth_tax / row['price']
                    shares -= shares_to_sell
                    portfolio_value = shares * row['price']
            
            # Set year start value for next year
            if row['is_year_end']:
                year_start_value = portfolio_value
            
            # Monthly contribution at month end
            if row['is_month_end'] and not (row['year'] == 2021 and row['month'] == 1):
                new_shares = monthly_contrib / row['price']
                shares += new_shares
                portfolio_value = shares * row['price']
                if row['is_year_end']:
                    year_start_value = portfolio_value
            
            results.append({
                'date': row['date'],
                'portfolio_value': portfolio_value,
                'cumulative_tax': cumulative_tax,
                'cumulative_income_tax': cumulative_income_tax,
                'cumulative_wealth_tax': cumulative_wealth_tax,
                'shares': shares,
                'loss_carryforward': loss_carryforward
            })
        
        return pd.DataFrame(results)
    
    # Run simulations
    with st.spinner("Running simulations..."):
        forfait_results = simulate_forfait(df, initial_investment, monthly_contribution, 
                                           forfait_r, forfait_tax_rate)
        accrual_results = simulate_accrual(df, initial_investment, monthly_contribution, 
                                          accrual_tax_rate)
        labor_results = simulate_labor(df, initial_investment, monthly_contribution, 
                                      labor_tax_rate, wealth_tax_rate, wealth_tax_threshold)
    
    # Display key metrics
    st.header("📈 Final Results Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Forfait Model", 
                 f"€{forfait_results.iloc[-1]['portfolio_value']:,.0f}",
                 f"Tax: €{forfait_results.iloc[-1]['cumulative_tax']:,.0f}")
    
    with col2:
        st.metric("Accrual Model", 
                 f"€{accrual_results.iloc[-1]['portfolio_value']:,.0f}",
                 f"Tax: €{accrual_results.iloc[-1]['cumulative_tax']:,.0f}")
    
    with col3:
        st.metric("Labor Model", 
                 f"€{labor_results.iloc[-1]['portfolio_value']:,.0f}",
                 f"Tax: €{labor_results.iloc[-1]['cumulative_tax']:,.0f}")
    
    # Calculate total contributions
    num_months = len(df[df['is_month_end']]) - 1  # Exclude first month
    total_contributed = initial_investment + (monthly_contribution * num_months)
    
    st.info(f"💰 Total Contributed: €{total_contributed:,.0f} over {num_months} months")
    
    # Comparison table
    st.header("📊 Detailed Comparison")
    
    comparison_data = {
        'Model': ['Forfait', 'Accrual', 'Labor'],
        'Final Portfolio Value': [
            f"€{forfait_results.iloc[-1]['portfolio_value']:,.0f}",
            f"€{accrual_results.iloc[-1]['portfolio_value']:,.0f}",
            f"€{labor_results.iloc[-1]['portfolio_value']:,.0f}"
        ],
        'Total Tax Paid': [
            f"€{forfait_results.iloc[-1]['cumulative_tax']:,.0f}",
            f"€{accrual_results.iloc[-1]['cumulative_tax']:,.0f}",
            f"€{labor_results.iloc[-1]['cumulative_tax']:,.0f}"
        ],
        'Net Gain': [
            f"€{forfait_results.iloc[-1]['portfolio_value'] - total_contributed:,.0f}",
            f"€{accrual_results.iloc[-1]['portfolio_value'] - total_contributed:,.0f}",
            f"€{labor_results.iloc[-1]['portfolio_value'] - total_contributed:,.0f}"
        ],
        'Effective Tax Rate': [
            f"{(forfait_results.iloc[-1]['cumulative_tax'] / forfait_results.iloc[-1]['portfolio_value']) * 100:.2f}%",
            f"{(accrual_results.iloc[-1]['cumulative_tax'] / accrual_results.iloc[-1]['portfolio_value']) * 100:.2f}%",
            f"{(labor_results.iloc[-1]['cumulative_tax'] / labor_results.iloc[-1]['portfolio_value']) * 100:.2f}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # Visualizations
    st.header("📉 Portfolio Value Over Time")
    
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=forfait_results['date'],
        y=forfait_results['portfolio_value'],
        name='Forfait',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig1.add_trace(go.Scatter(
        x=accrual_results['date'],
        y=accrual_results['portfolio_value'],
        name='Accrual',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig1.add_trace(go.Scatter(
        x=labor_results['date'],
        y=labor_results['portfolio_value'],
        name='Labor',
        line=dict(color='#95E1D3', width=2)
    ))
    
    fig1.update_layout(
        title='After-Tax Portfolio Value Comparison',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (EUR)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Cumulative Tax Chart
    st.header("💸 Cumulative Tax Paid")
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=forfait_results['date'],
        y=forfait_results['cumulative_tax'],
        name='Forfait',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy'
    ))
    
    fig2.add_trace(go.Scatter(
        x=accrual_results['date'],
        y=accrual_results['cumulative_tax'],
        name='Accrual',
        line=dict(color='#4ECDC4', width=2),
        fill='tozeroy'
    ))
    
    fig2.add_trace(go.Scatter(
        x=labor_results['date'],
        y=labor_results['cumulative_tax'],
        name='Labor',
        line=dict(color='#95E1D3', width=2),
        fill='tozeroy'
    ))
    
    fig2.update_layout(
        title='Cumulative Tax Paid Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Tax (EUR)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Side by side comparison
    st.header("⚖️ Final Value Comparison")
    
    fig3 = go.Figure(data=[
        go.Bar(name='Portfolio Value', 
               x=['Forfait', 'Accrual', 'Labor'],
               y=[forfait_results.iloc[-1]['portfolio_value'],
                  accrual_results.iloc[-1]['portfolio_value'],
                  labor_results.iloc[-1]['portfolio_value']],
               marker_color=['#FF6B6B', '#4ECDC4', '#95E1D3']),
        go.Bar(name='Tax Paid', 
               x=['Forfait', 'Accrual', 'Labor'],
               y=[forfait_results.iloc[-1]['cumulative_tax'],
                  accrual_results.iloc[-1]['cumulative_tax'],
                  labor_results.iloc[-1]['cumulative_tax']],
               marker_color=['#C44569', '#1B9AAA', '#38A3A5'])
    ])
    
    fig3.update_layout(
        barmode='group',
        title='Final Portfolio Value vs Total Tax Paid',
        yaxis_title='Amount (EUR)',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Additional insights for Labor model
    if labor_results.iloc[-1]['cumulative_wealth_tax'] > 0:
        st.header("🏛️ Labor Model Tax Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Income Tax (on gains)", 
                     f"€{labor_results.iloc[-1]['cumulative_income_tax']:,.0f}")
        with col2:
            st.metric("Wealth Tax", 
                     f"€{labor_results.iloc[-1]['cumulative_wealth_tax']:,.0f}")
    
    # Download data option
    st.header("💾 Download Results")
    
    # Combine results for download
    download_df = pd.DataFrame({
        'Date': forfait_results['date'],
        'Forfait_Portfolio': forfait_results['portfolio_value'],
        'Forfait_Tax': forfait_results['cumulative_tax'],
        'Accrual_Portfolio': accrual_results['portfolio_value'],
        'Accrual_Tax': accrual_results['cumulative_tax'],
        'Labor_Portfolio': labor_results['portfolio_value'],
        'Labor_Tax': labor_results['cumulative_tax']
    })
    
    csv = download_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="tax_model_comparison.csv",
        mime="text/csv"
    )

else:
    st.error("Unable to load data. Please check your internet connection.")
