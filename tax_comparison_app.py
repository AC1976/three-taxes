import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Investment Tax Model Comparison", layout="wide")

st.title("🏦 Investment Tax Model Comparison")
st.markdown("Compare three different taxation models on investment returns using S&P 500 (SPY) data")

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

st.sidebar.markdown("---")
run_button = st.sidebar.button("🔄 Recalculate", type="primary", use_container_width=True)

# Hardcoded SPY monthly data (2001-2024) - End of month adjusted close prices
# Source: Historical SPY data
SPY_DATA = {
    '2001-01-31': 128.82, '2001-02-28': 119.77, '2001-03-30': 113.49, '2001-04-30': 121.61,
    '2001-05-31': 121.88, '2001-06-29': 119.53, '2001-07-31': 117.93, '2001-08-31': 112.52,
    '2001-09-28': 103.30, '2001-10-31': 107.48, '2001-11-30': 114.28, '2001-12-31': 115.36,
    '2002-01-31': 112.58, '2002-02-28': 110.67, '2002-03-28': 114.50, '2002-04-30': 109.14,
    '2002-05-31': 107.85, '2002-06-28': 100.02, '2002-07-31': 94.18, '2002-08-30': 95.48,
    '2002-09-30': 86.62, '2002-10-31': 92.64, '2002-11-29': 96.73, '2002-12-31': 92.93,
    '2003-01-31': 90.53, '2003-02-28': 88.77, '2003-03-31': 89.83, '2003-04-30': 95.40,
    '2003-05-30': 99.88, '2003-06-30': 100.19, '2003-07-31': 102.58, '2003-08-29': 104.05,
    '2003-09-30': 103.57, '2003-10-31': 108.24, '2003-11-28': 108.98, '2003-12-31': 111.23,
    '2004-01-30': 113.71, '2004-02-27': 115.73, '2004-03-31': 114.55, '2004-04-30': 114.34,
    '2004-05-28': 115.37, '2004-06-30': 116.84, '2004-07-30': 112.38, '2004-08-31': 113.15,
    '2004-09-30': 113.27, '2004-10-29': 115.30, '2004-11-30': 119.67, '2004-12-31': 121.58,
    '2005-01-31': 119.38, '2005-02-28': 121.54, '2005-03-31': 119.17, '2005-04-29': 117.93,
    '2005-05-31': 120.73, '2005-06-30': 120.19, '2005-07-29': 123.56, '2005-08-31': 122.26,
    '2005-09-30': 123.63, '2005-10-31': 121.82, '2005-11-30': 125.40, '2005-12-30': 125.36,
    '2006-01-31': 128.87, '2006-02-28': 128.73, '2006-03-31': 130.00, '2006-04-28': 131.24,
    '2006-05-31': 128.04, '2006-06-30': 128.15, '2006-07-31': 128.28, '2006-08-31': 131.47,
    '2006-09-29': 133.96, '2006-10-31': 137.81, '2006-11-30': 139.70, '2006-12-29': 141.60,
    '2007-01-31': 141.97, '2007-02-28': 142.19, '2007-03-30': 143.76, '2007-04-30': 148.36,
    '2007-05-31': 151.88, '2007-06-29': 149.96, '2007-07-31': 146.14, '2007-08-31': 146.50,
    '2007-09-28': 151.68, '2007-10-31': 153.48, '2007-11-30': 147.05, '2007-12-31': 146.21,
    '2008-01-31': 135.07, '2008-02-29': 136.00, '2008-03-31': 132.65, '2008-04-30': 139.49,
    '2008-05-30': 140.28, '2008-06-30': 131.19, '2008-07-31': 126.39, '2008-08-29': 129.37,
    '2008-09-30': 116.77, '2008-10-31': 101.48, '2008-11-28': 90.24, '2008-12-31': 90.24,
    '2009-01-30': 82.83, '2009-02-27': 73.93, '2009-03-31': 79.73, '2009-04-30': 87.24,
    '2009-05-29': 92.53, '2009-06-30': 91.99, '2009-07-31': 98.06, '2009-08-31': 101.51,
    '2009-09-30': 106.72, '2009-10-30': 103.56, '2009-11-30': 110.99, '2009-12-31': 111.44,
    '2010-01-29': 107.39, '2010-02-26': 110.74, '2010-03-31': 117.64, '2010-04-30': 118.81,
    '2010-05-28': 109.43, '2010-06-30': 103.27, '2010-07-30': 112.77, '2010-08-31': 107.64,
    '2010-09-30': 114.59, '2010-10-29': 118.81, '2010-11-30': 119.41, '2010-12-31': 125.75,
    '2011-01-31': 128.27, '2011-02-28': 132.18, '2011-03-31': 132.47, '2011-04-29': 135.14,
    '2011-05-31': 133.42, '2011-06-30': 130.84, '2011-07-29': 127.49, '2011-08-31': 120.92,
    '2011-09-30': 112.77, '2011-10-31': 124.60, '2011-11-30': 125.75, '2011-12-30': 126.24,
    '2012-01-31': 131.89, '2012-02-29': 136.05, '2012-03-30': 139.84, '2012-04-30': 137.91,
    '2012-05-31': 133.13, '2012-06-29': 136.30, '2012-07-31': 138.54, '2012-08-31': 140.99,
    '2012-09-28': 143.97, '2012-10-31': 142.11, '2012-11-30': 142.71, '2012-12-31': 142.41,
    '2013-01-31': 150.20, '2013-02-28': 151.08, '2013-03-28': 156.73, '2013-04-30': 158.98,
    '2013-05-31': 164.85, '2013-06-28': 162.73, '2013-07-31': 169.30, '2013-08-30': 164.11,
    '2013-09-30': 168.10, '2013-10-31': 175.77, '2013-11-29': 179.97, '2013-12-31': 184.69,
    '2014-01-31': 177.72, '2014-02-28': 185.47, '2014-03-31': 186.98, '2014-04-30': 188.23,
    '2014-05-30': 192.46, '2014-06-30': 195.61, '2014-07-31': 194.72, '2014-08-29': 200.07,
    '2014-09-30': 198.84, '2014-10-31': 205.00, '2014-11-28': 208.92, '2014-12-31': 205.54,
    '2015-01-30': 199.44, '2015-02-27': 211.00, '2015-03-31': 207.40, '2015-04-30': 209.84,
    '2015-05-29': 211.90, '2015-06-30': 207.84, '2015-07-31': 209.58, '2015-08-31': 198.84,
    '2015-09-30': 192.66, '2015-10-30': 210.20, '2015-11-30': 210.59, '2015-12-31': 203.87,
    '2016-01-29': 194.02, '2016-02-29': 193.00, '2016-03-31': 206.33, '2016-04-29': 207.40,
    '2016-05-31': 211.04, '2016-06-30': 211.70, '2016-07-29': 217.12, '2016-08-31': 217.38,
    '2016-09-30': 216.30, '2016-10-31': 214.99, '2016-11-30': 220.38, '2016-12-30': 223.53,
    '2017-01-31': 227.76, '2017-02-28': 236.47, '2017-03-31': 236.37, '2017-04-28': 238.97,
    '2017-05-31': 241.80, '2017-06-30': 243.43, '2017-07-31': 248.41, '2017-08-31': 247.52,
    '2017-09-29': 250.33, '2017-10-31': 255.67, '2017-11-30': 263.13, '2017-12-29': 266.86,
    '2018-01-31': 281.18, '2018-02-28': 272.98, '2018-03-29': 265.10, '2018-04-30': 266.60,
    '2018-05-31': 272.38, '2018-06-29': 272.13, '2018-07-31': 281.84, '2018-08-31': 289.81,
    '2018-09-28': 290.89, '2018-10-31': 268.09, '2018-11-30': 278.81, '2018-12-31': 249.92,
    '2019-01-31': 270.47, '2019-02-28': 279.35, '2019-03-29': 283.40, '2019-04-30': 293.87,
    '2019-05-31': 276.74, '2019-06-28': 295.87, '2019-07-31': 300.09, '2019-08-30': 293.47,
    '2019-09-30': 296.77, '2019-10-31': 304.70, '2019-11-29': 314.08, '2019-12-31': 321.86,
    '2020-01-31': 321.34, '2020-02-28': 298.77, '2020-03-31': 254.87, '2020-04-30': 289.63,
    '2020-05-29': 304.02, '2020-06-30': 306.65, '2020-07-31': 322.17, '2020-08-31': 351.88,
    '2020-09-30': 335.48, '2020-10-30': 326.54, '2020-11-30': 362.75, '2020-12-31': 370.07,
    '2021-01-29': 370.83, '2021-02-26': 380.36, '2021-03-31': 395.62, '2021-04-30': 416.98,
    '2021-05-28': 418.83, '2021-06-30': 429.11, '2021-07-30': 439.29, '2021-08-31': 449.37,
    '2021-09-30': 429.26, '2021-10-29': 458.77, '2021-11-30': 453.05, '2021-12-31': 474.96,
    '2022-01-31': 441.61, '2022-02-28': 435.28, '2022-03-31': 454.34, '2022-04-29': 413.25,
    '2022-05-31': 414.53, '2022-06-30': 379.44, '2022-07-29': 408.49, '2022-08-31': 395.25,
    '2022-09-30': 357.04, '2022-10-31': 386.22, '2022-11-30': 398.11, '2022-12-30': 382.46,
    '2023-01-31': 401.03, '2023-02-28': 394.33, '2023-03-31': 407.77, '2023-04-28': 410.91,
    '2023-05-31': 420.55, '2023-06-30': 441.41, '2023-07-31': 448.68, '2023-08-31': 448.48,
    '2023-09-29': 429.38, '2023-10-31': 420.99, '2023-11-30': 450.52, '2023-12-29': 471.63,
    '2024-01-31': 484.98, '2024-02-29': 509.80, '2024-03-28': 523.95, '2024-04-30': 503.48,
    '2024-05-31': 526.33, '2024-06-28': 544.47, '2024-07-31': 549.96, '2024-08-30': 563.07,
    '2024-09-30': 573.37, '2024-10-31': 573.39
}

st.sidebar.success(f"✓ Loaded {len(SPY_DATA)} months of SPY data")

# Convert to DataFrame
df = pd.DataFrame(list(SPY_DATA.items()), columns=['date', 'price'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Calculate monthly returns (use forward fill for intermediate days)
df['monthly_return'] = df['price'].pct_change()

# Create month-end flags
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['is_month_end'] = True  # All our data points are month-end
df['is_year_end'] = df['date'].dt.month == 12

# Mark first data point of each year (for forfait tax)
df['is_first_trading_day'] = df.groupby('year')['date'].transform('min') == df['date']

# Initialize tracking dataframes
def simulate_forfait(df, initial, monthly_contrib, r_rate, tax_rate):
    """Simulate Forfait model"""
    portfolio_value = initial
    shares = initial / df.iloc[0]['price']
    cumulative_tax = 0
    
    results = []
    
    for idx, row in df.iterrows():
        # Apply monthly return to shares
        if idx > 0:
            shares = shares * (1 + row['monthly_return'])
        
        portfolio_value = shares * row['price']
        
        # Annual tax on first data point of year (based on portfolio value)
        if row['is_first_trading_day'] and row['year'] > 2001:
            tax_base = portfolio_value
            tax_amount = tax_base * (r_rate / 100) * (tax_rate / 100)
            cumulative_tax += tax_amount
            # Pay tax by reducing shares
            shares_to_sell = tax_amount / row['price']
            shares -= shares_to_sell
            portfolio_value = shares * row['price']
        
        # Monthly contribution at month end
        if row['is_month_end'] and not (row['year'] == 2001 and row['month'] == 1):
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
        # Apply monthly return
        if idx > 0:
            shares = shares * (1 + row['monthly_return'])
        
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
        if row['is_month_end'] and not (row['year'] == 2001 and row['month'] == 1):
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
        # Apply monthly return
        if idx > 0:
            shares = shares * (1 + row['monthly_return'])
        
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
        if row['is_month_end'] and not (row['year'] == 2001 and row['month'] == 1):
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
with st.spinner("Running simulations with current parameters..."):
    forfait_results = simulate_forfait(df, initial_investment, monthly_contribution, 
                                       forfait_r, forfait_tax_rate)
    accrual_results = simulate_accrual(df, initial_investment, monthly_contribution, 
                                      accrual_tax_rate)
    labor_results = simulate_labor(df, initial_investment, monthly_contribution, 
                                  labor_tax_rate, wealth_tax_rate, wealth_tax_threshold)

# Show that calculation is complete
st.success("✓ Calculations complete!")

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
num_months = len(df) - 1  # Exclude first month
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
