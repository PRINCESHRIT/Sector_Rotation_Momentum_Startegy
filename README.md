# Sector_Rotation_Momentum_Startegy

Sector Rotation Strategy: Dynamic allocation strategy for Indian stock market sectors using momentum, trend strength, volume, and risk management;<br>
Features: Calculates metrics (momentum, trend strength, volatility, relative strength), generates signals (BUY, SELL, HOLD), and manages risk through configurable limits;<br>
Dependencies: pandas, numpy, ta-lib, json, logging, dataclasses (install via pip install pandas numpy ta-lib);<br>
Usage: Import SectorRotationStrategy, prepare sector data as DataFrames (OHLCV format), run execute_rotation for allocations, and get_signals for trading actions;<br>
Key Configuration: Defined sectors, risk weights, and momentum thresholds in config.json; enforced diversification with allocation limits (max_sector, min_sectors, min_allocation);<br>
Example Output: { "NIFTY_BANK": 0.30, "NIFTY_IT": 0.25, "NIFTY_PHARMA": 0.20 };<br>
Outputs: Sector allocations, trading signals, and comprehensive logs in sector_rotation.log;<br>
Future Enhancements: Add real-time data APIs, machine learning models for prediction, and macroeconomic factor integration.<br>
