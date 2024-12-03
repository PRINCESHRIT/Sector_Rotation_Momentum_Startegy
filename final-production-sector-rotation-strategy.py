from typing import Dict, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import talib as ta
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SectorMetrics:
    """Container for sector analysis metrics"""
    momentum_score: float
    trend_score: float
    volume_score: float
    volatility: float
    relative_strength: float
    risk_score: float

class SectorRotationStrategy:
    """Production-ready Sector Rotation Strategy for Indian Markets"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize strategy with configuration
        
        Args:
            config_path: Path to configuration JSON file
        """
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._init_components()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sector_rotation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SectorRotation')

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load strategy configuration"""
        default_config = {
            'sectors': {
                'NIFTY_BANK': {'risk_weight': 1.2, 'min_momentum': 0.3},
                'NIFTY_IT': {'risk_weight': 1.1, 'min_momentum': 0.25},
                'NIFTY_AUTO': {'risk_weight': 1.0, 'min_momentum': 0.2},
                'NIFTY_PHARMA': {'risk_weight': 0.9, 'min_momentum': 0.2},
                'NIFTY_FMCG': {'risk_weight': 0.8, 'min_momentum': 0.15}
            },
            'position_limits': {
                'max_sector': 0.35,
                'min_sectors': 2,
                'max_sectors': 4,
                'min_allocation': 0.1
            },
            'risk_params': {
                'max_drawdown': -0.05,
                'volatility_threshold': 0.20,
                'correlation_threshold': 0.7
            },
            'technical_params': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'volume_ma': 20,
                'trend_period': 50
            }
        }
        
        try:
            if config_path:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            return default_config
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return default_config

    def _init_components(self):
        """Initialize strategy components"""
        self.metrics_cache = {}
        self.current_positions = {}
        self.performance_metrics = []

    def calculate_sector_metrics(self, data: pd.DataFrame, sector: str) -> SectorMetrics:
        """
        Calculate comprehensive sector metrics
        
        Args:
            data: DataFrame with OHLCV data
            sector: Sector name
            
        Returns:
            SectorMetrics object
        """
        try:
            # Validate input data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Required: {required_cols}")
            
            # Calculate momentum score
            momentum = self._calculate_momentum(data)
            
            # Calculate trend score
            trend = self._calculate_trend_strength(data)
            
            # Calculate volume score
            volume = self._calculate_volume_score(data)
            
            # Calculate risk metrics
            volatility = self._calculate_volatility(data)
            rel_strength = self._calculate_relative_strength(data)
            risk_score = self._calculate_risk_score(data)
            
            return SectorMetrics(
                momentum_score=momentum,
                trend_score=trend,
                volume_score=volume,
                volatility=volatility,
                relative_strength=rel_strength,
                risk_score=risk_score
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {sector}: {str(e)}")
            raise

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum using multiple indicators"""
        try:
            close = data['close']
            
            # RSI
            rsi = ta.RSI(close, timeperiod=self.config['technical_params']['rsi_period'])
            
            # MACD
            macd, signal, _ = ta.MACD(
                close, 
                fastperiod=self.config['technical_params']['macd_fast'],
                slowperiod=self.config['technical_params']['macd_slow'],
                signalperiod=self.config['technical_params']['macd_signal']
            )
            
            # Price ROC
            roc = ta.ROC(close, timeperiod=self.config['technical_params']['trend_period'])
            
            # Combine signals
            momentum = (
                0.4 * (rsi.iloc[-1] - 50) / 50 +
                0.3 * np.sign(macd.iloc[-1] - signal.iloc[-1]) +
                0.3 * roc.iloc[-1] / 100
            )
            
            return float(np.clip(momentum, -1, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {str(e)}")
            raise

    def execute_rotation(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Execute sector rotation strategy
        
        Args:
            sector_data: Dict mapping sector names to their OHLCV DataFrames
            
        Returns:
            Dict mapping sectors to their target allocations
        """
        try:
            # Validate input data
            if not sector_data:
                raise ValueError("Empty sector data provided")
            
            # Calculate metrics for each sector
            sector_metrics = {}
            for sector, data in sector_data.items():
                metrics = self.calculate_sector_metrics(data, sector)
                sector_metrics[sector] = metrics
            
            # Score and rank sectors
            sector_scores = self._rank_sectors(sector_metrics)
            
            # Apply position sizing
            allocations = self._calculate_allocations(sector_scores, sector_metrics)
            
            # Apply risk overlay
            final_allocations = self._apply_risk_overlay(allocations, sector_metrics)
            
            # Validate final allocations
            self._validate_allocations(final_allocations)
            
            return final_allocations
            
        except Exception as e:
            self.logger.error(f"Error executing rotation: {str(e)}")
            raise

    def _rank_sectors(self, metrics: Dict[str, SectorMetrics]) -> Dict[str, float]:
        """Rank sectors based on combined metrics"""
        try:
            scores = {}
            for sector, metric in metrics.items():
                # Combined score with risk adjustment
                score = (
                    0.4 * metric.momentum_score +
                    0.3 * metric.trend_score +
                    0.2 * metric.volume_score -
                    0.1 * metric.risk_score
                )
                scores[sector] = score
            
            return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            self.logger.error(f"Error ranking sectors: {str(e)}")
            raise

    def _calculate_allocations(
        self, 
        scores: Dict[str, float], 
        metrics: Dict[str, SectorMetrics]
    ) -> Dict[str, float]:
        """Calculate sector allocations"""
        try:
            allocations = {}
            total_score = sum(scores.values())
            
            for sector, score in scores.items():
                if score > self.config['sectors'][sector]['min_momentum']:
                    # Base allocation
                    allocation = score / total_score
                    
                    # Risk-adjust allocation
                    risk_adj = 1.0 - metrics[sector].risk_score
                    allocation *= risk_adj
                    
                    # Apply position limits
                    allocation = min(
                        allocation,
                        self.config['position_limits']['max_sector']
                    )
                    
                    if allocation >= self.config['position_limits']['min_allocation']:
                        allocations[sector] = allocation
            
            return self._normalize_allocations(allocations)
            
        except Exception as e:
            self.logger.error(f"Error calculating allocations: {str(e)}")
            raise

    def _normalize_allocations(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Normalize allocations to sum to 1.0"""
        total = sum(allocations.values())
        if total > 0:
            return {k: v/total for k, v in allocations.items()}
        return allocations

    def _validate_allocations(self, allocations: Dict[str, float]):
        """Validate final allocations"""
        try:
            total_allocation = sum(allocations.values())
            if not (0.99 <= total_allocation <= 1.01):
                raise ValueError(f"Invalid total allocation: {total_allocation}")
            
            if len(allocations) < self.config['position_limits']['min_sectors']:
                raise ValueError(f"Insufficient sector diversification: {len(allocations)}")
                
        except Exception as e:
            self.logger.error(f"Allocation validation failed: {str(e)}")
            raise

    def get_signals(self, allocations: Dict[str, float]) -> Dict[str, str]:
        """Generate trading signals from allocations"""
        signals = {}
        for sector, allocation in allocations.items():
            current = self.current_positions.get(sector, 0.0)
            
            if allocation > current:
                signals[sector] = 'BUY'
            elif allocation < current:
                signals[sector] = 'SELL'
            else:
                signals[sector] = 'HOLD'
                
        return signals

    def update_positions(self, allocations: Dict[str, float]):
        """Update current positions"""
        self.current_positions = allocations.copy()
