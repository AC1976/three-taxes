# Investment Tax Model Comparison Tool

This Streamlit application compares three different taxation models for investment returns using MSCI World Index data from 2021-2025.

## Models Compared

1. **Forfait Model**: Fixed deemed return rate with annual taxation
2. **Accrual Model**: Mark-to-market taxation with loss carryforward
3. **Labor Model**: Higher rate mark-to-market taxation plus wealth tax

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the App

Run the following command in your terminal:
```bash
streamlit run tax_comparison_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Features

- **Interactive Parameters**: Adjust all tax rates, thresholds, and investment amounts
- **Real-time Data**: Fetches actual MSCI World Index performance data
- **Visual Comparisons**: 
  - Portfolio value over time
  - Cumulative taxes paid
  - Side-by-side final value comparison
- **Detailed Metrics**: Final values, total taxes, net gains, and effective tax rates
- **Export**: Download results as CSV for further analysis

## Default Settings

- Initial Investment: €100,000
- Monthly Contribution: €1,000
- Forfait: 4% deemed return, 36% tax rate
- Accrual: 36% tax rate
- Labor: 55% tax rate, 1% wealth tax on amounts > €200,000

All parameters can be adjusted via the sidebar!
