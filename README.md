# Sector_Rotation_Momentum_Startegy
Market's DJ: Dropping the beat at the right time and sector;
Sector Rotation Strategy: Dynamic allocation strategy for Indian stock market sectors using momentum, trend strength, volume, and risk management;
Features: Calculates metrics (momentum, trend strength, volatility, relative strength), generates signals (BUY, SELL, HOLD), and manages risk through configurable limits;
Dependencies: pandas, numpy, ta-lib, json, logging, dataclasses (install via pip install pandas numpy ta-lib);
Usage: Import SectorRotationStrategy, prepare sector data as DataFrames (OHLCV format), run execute_rotation for allocations, and get_signals for trading actions;
Key Configuration: Defined sectors, risk weights, and momentum thresholds in config.json; enforced diversification with allocation limits (max_sector, min_sectors, min_allocation);
Example Output: { "NIFTY_BANK": 0.30, "NIFTY_IT": 0.25, "NIFTY_PHARMA": 0.20 };
Outputs: Sector allocations, trading signals, and comprehensive logs in sector_rotation.log;
Future Enhancements: Add real-time data APIs, machine learning models for prediction, and macroeconomic factor integration.
