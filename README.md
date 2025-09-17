# End-of-Month Strategy Analysis

An analysis of end-of-month market timing strategies based on academic research by Etula et al. (2019) and Hartley et al. (2020). The premise is that the liquidity demand patterns that occur before month-end are due to the need for market participants to raise cash to meet payment obligations.   The implementation is in python.

## Overview

This project analyzes out-of-sample performance of end-of-month (EOM) market timing strategies for equities (via SPY) and bonds (via 10y UST).  The analysis identifies patterns in market returns based on distance from month-end and calculates statistical significance of potential trading strategies.

## Key Features

- **Excess Returns Calculation**: Computes risk-adjusted returns for equity and bond series
- **EOM Distance Analysis**: Analyzes returns by number of days from end-of-month
- **Statistical Significance**: Calculates Sharpe ratios and tests for different EOM windows
- **Comprehensive Reporting**: Generates multi-page PDF reports with detailed visualizations
- **Data Export**: Exports intermediate results to CSV for further analysis

## Project Structure

```
├── src/
│   └── end-of-month-strategy-analysis.py    # Main analysis script
├── doc/
│   ├── end_of_month_strategy_analysis.pdf   # Analysis documentation
│   ├── Etula_Dash_For_Cash_Slides.pdf      # Reference slides
│   ├── etula2019_Dash for Cash...pdf       # Original research paper
│   └── Hartley_Predictable_End...pdf       # Bond strategy research
└── .gitignore
```

## Output Files

The analysis generates the following output files:

- `{series_name}_{start_date}_{end_date}.pdf`: Multi-page analysis report with plots
- `{series_name}_{start_date}_{end_date}.csv`: Detailed intermediate results and statistics

## Usage

Run the main analysis script:

```bash
python src/end-of-month-strategy-analysis.py
```

## Academic References

We based our analysis on the following research:

1. **Etula et al. (2019)**: "Dash for Cash: Monthly Market Impact of Institutional Liquidity Needs" - Equity market analysis
2. **Hartley et al. (2020)**: "Predictable End-of-Month Treasury Returns" - Bond market analysis

## Dependencies

- numpy, pandas, matplotlib, scipy

## Configuration

The script includes configuration options:
- `short_rate_series_name`: Treasury rate series identifier (default: 'H15T3M')
- `DEBUG_MODE`: Enable/disable debug output (default: False)
