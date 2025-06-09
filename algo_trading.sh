#!/bin/bash

# Crypto Trading Feature Generator for Existing Projects
# Modular, expandable, and focused on trading capabilities

set -euo pipefail

# ============================================================================
# CONFIGURATION & GLOBALS
# ============================================================================

readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_NAME="Trading Feature Generator"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Project info
PROJECT_NAME=$(basename "$PWD")
declare -A PROJECT_CONFIG
PROJECT_CONFIG[name]="$PROJECT_NAME"

# Feature registry - easily expandable
declare -A TRADING_FEATURES
declare -A FEATURE_DEPENDENCIES
declare -A FEATURE_DESCRIPTIONS

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

safe_create_file() {
    local file_path="$1"
    local content="$2"
    local force="${3:-false}"
    
    if [ -f "$file_path" ] && [ "$force" != "true" ]; then
        log_warning "File $file_path exists. Use --force to overwrite"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && return 1
    fi
    
    mkdir -p "$(dirname "$file_path")"
    echo "$content" > "$file_path"
    log_success "Created: $file_path"
}

safe_create_dir() {
    local dir_path="$1"
    [ -d "$dir_path" ] || mkdir -p "$dir_path"
    log_info "Directory ready: $dir_path"
}

check_feature_exists() {
    local feature="$1"
    case $feature in
        exchange_api) [ -d "src/${PROJECT_NAME}/exchanges" ] ;;
        strategies) [ -d "src/${PROJECT_NAME}/strategies" ] ;;
        backtesting) [ -f "src/${PROJECT_NAME}/backtesting.py" ] ;;
        data_feeds) [ -f "src/${PROJECT_NAME}/data_feeds.py" ] ;;
        notifications) [ -f "src/${PROJECT_NAME}/notifications.py" ] ;;
        risk_management) [ -f "src/${PROJECT_NAME}/risk_manager.py" ] ;;
        portfolio) [ -f "src/${PROJECT_NAME}/portfolio.py" ] ;;
        paper_trading) [ -f "src/${PROJECT_NAME}/paper_trading.py" ] ;;
        live_trading) [ -f "src/${PROJECT_NAME}/live_trading.py" ] ;;
        monitoring) [ -f "src/${PROJECT_NAME}/monitoring.py" ] ;;
        *) false ;;
    esac
}

# ============================================================================
# FEATURE REGISTRY SETUP
# ============================================================================

setup_feature_registry() {
    # Core Trading Features
    TRADING_FEATURES[exchange_api]="Exchange API Integration"
    TRADING_FEATURES[strategies]="Trading Strategy Framework"
    TRADING_FEATURES[backtesting]="Backtesting Engine"
    TRADING_FEATURES[data_feeds]="Real-time Data Feeds"
    TRADING_FEATURES[notifications]="Alert & Notification System"
    TRADING_FEATURES[risk_management]="Risk Management System"
    TRADING_FEATURES[portfolio]="Portfolio Management"
    TRADING_FEATURES[paper_trading]="Paper Trading Simulator"
    TRADING_FEATURES[live_trading]="Live Trading Engine"
    TRADING_FEATURES[monitoring]="Performance Monitoring"
    
    # Feature Dependencies
    FEATURE_DEPENDENCIES[exchange_api]="aiohttp>=3.8.0 ccxt>=3.0.0"
    FEATURE_DEPENDENCIES[strategies]="pandas>=1.5.0 numpy>=1.21.0 ta-lib>=0.4.0"
    FEATURE_DEPENDENCIES[backtesting]="pandas>=1.5.0 numpy>=1.21.0 matplotlib>=3.5.0"
    FEATURE_DEPENDENCIES[data_feeds]="websockets>=10.0 aiohttp>=3.8.0"
    FEATURE_DEPENDENCIES[notifications]="python-telegram-bot>=20.0 discord.py>=2.0"
    FEATURE_DEPENDENCIES[risk_management]="pandas>=1.5.0 numpy>=1.21.0"
    FEATURE_DEPENDENCIES[portfolio]="pandas>=1.5.0 numpy>=1.21.0"
    FEATURE_DEPENDENCIES[paper_trading]="pandas>=1.5.0"
    FEATURE_DEPENDENCIES[live_trading]="ccxt>=3.0.0 aiohttp>=3.8.0"
    FEATURE_DEPENDENCIES[monitoring]="prometheus-client>=0.15.0 grafana-api>=1.0.3"
    
    # Feature Descriptions
    FEATURE_DESCRIPTIONS[exchange_api]="Connect to crypto exchanges (Binance, Bybit, etc.)"
    FEATURE_DESCRIPTIONS[strategies]="Technical analysis strategies (SMA, RSI, MACD, etc.)"
    FEATURE_DESCRIPTIONS[backtesting]="Test strategies on historical data"
    FEATURE_DESCRIPTIONS[data_feeds]="Real-time price feeds and market data"
    FEATURE_DESCRIPTIONS[notifications]="Telegram/Discord alerts for signals"
    FEATURE_DESCRIPTIONS[risk_management]="Position sizing, stop-loss, risk controls"
    FEATURE_DESCRIPTIONS[portfolio]="Track positions, P&L, performance metrics"
    FEATURE_DEPENDENCIES[paper_trading]="Simulate trading without real money"
    FEATURE_DESCRIPTIONS[live_trading]="Execute real trades automatically"
    FEATURE_DESCRIPTIONS[monitoring]="Dashboard and performance tracking"
}

# ============================================================================
# FEATURE IMPLEMENTATIONS
# ============================================================================

add_exchange_api_feature() {
    log_info "Adding Exchange API Integration..."
    
    safe_create_dir "src/${PROJECT_NAME}/exchanges"
    
    # Base Exchange Interface
    safe_create_file "src/${PROJECT_NAME}/exchanges/__init__.py" '"""Exchange API integrations."""'
    
    safe_create_file "src/${PROJECT_NAME}/exchanges/base_exchange.py" '"""Base exchange interface."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import asyncio
import logging

logger = logging.getLogger(__name__)

class BaseExchange(ABC):
    """Abstract base class for all exchange implementations."""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.name = self.__class__.__name__.replace("Exchange", "").lower()
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    @abstractmethod
    async def connect(self):
        """Initialize connection to exchange."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to exchange."""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for symbol."""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get orderbook data."""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, amount: float, 
                         price: Optional[float] = None, order_type: str = "market") -> Dict[str, Any]:
        """Place an order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        pass
    
    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical kline/candlestick data."""
        pass
    
    def validate_symbol(self, symbol: str) -> str:
        """Validate and format symbol."""
        return symbol.upper().replace("/", "")
    
    def format_amount(self, amount: float, precision: int = 8) -> float:
        """Format amount to exchange precision."""
        return round(amount, precision)
'
    
    # Binance Implementation
    safe_create_file "src/${PROJECT_NAME}/exchanges/binance_exchange.py" '"""Binance exchange implementation."""
import asyncio
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any
import aiohttp
from .base_exchange import BaseExchange
import logging

logger = logging.getLogger(__name__)

class BinanceExchange(BaseExchange):
    """Binance exchange implementation with full API support."""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
        self.ws_url = "wss://testnet.binance.vision/ws" if testnet else "wss://stream.binance.com:9443/ws"
    
    async def connect(self):
        """Initialize aiohttp session."""
        self.session = aiohttp.ClientSession()
        logger.info(f"Connected to Binance {'testnet' if self.testnet else 'mainnet'}")
    
    async def disconnect(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            logger.info("Disconnected from Binance")
    
    def _generate_signature(self, params: str) -> str:
        """Generate API signature."""
        return hmac.new(
            self.api_secret.encode("utf-8"),
            params.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make authenticated request to Binance API."""
        if not self.session:
            await self.connect()
        
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if signed else {}
        
        if signed and params:
            params["timestamp"] = int(time.time() * 1000)
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            params["signature"] = self._generate_signature(query_string)
        
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                data = await response.json()
                if response.status != 200:
                    logger.error(f"Binance API error: {data}")
                    raise Exception(f"API Error: {data}")
                return data
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        data = await self._make_request("GET", "/api/v3/account", signed=True)
        balances = {}
        for balance in data.get("balances", []):
            if float(balance["free"]) > 0 or float(balance["locked"]) > 0:
                balances[balance["asset"]] = {
                    "free": float(balance["free"]),
                    "locked": float(balance["locked"]),
                    "total": float(balance["free"]) + float(balance["locked"])
                }
        return balances
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker statistics."""
        params = {"symbol": self.validate_symbol(symbol)}
        return await self._make_request("GET", "/api/v3/ticker/24hr", params)
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get order book."""
        params = {"symbol": self.validate_symbol(symbol), "limit": limit}
        return await self._make_request("GET", "/api/v3/depth", params)
    
    async def place_order(self, symbol: str, side: str, amount: float, 
                         price: Optional[float] = None, order_type: str = "MARKET") -> Dict[str, Any]:
        """Place an order."""
        params = {
            "symbol": self.validate_symbol(symbol),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": self.format_amount(amount)
        }
        
        if order_type.upper() == "LIMIT" and price:
            params["price"] = self.format_amount(price)
            params["timeInForce"] = "GTC"
        
        return await self._make_request("POST", "/api/v3/order", params, signed=True)
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order."""
        params = {"symbol": self.validate_symbol(symbol), "orderId": order_id}
        return await self._make_request("DELETE", "/api/v3/order", params, signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        params = {"symbol": self.validate_symbol(symbol)} if symbol else {}
        return await self._make_request("GET", "/api/v3/openOrders", params, signed=True)
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get kline/candlestick data."""
        params = {
            "symbol": self.validate_symbol(symbol),
            "interval": interval,
            "limit": limit
        }
        raw_data = await self._make_request("GET", "/api/v3/klines", params)
        
        # Format klines data
        klines = []
        for kline in raw_data:
            klines.append({
                "timestamp": int(kline[0]),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "close_time": int(kline[6]),
                "quote_volume": float(kline[7]),
                "trades": int(kline[8])
            })
        
        return klines
'
    
    # Exchange Factory
    safe_create_file "src/${PROJECT_NAME}/exchanges/factory.py" '"""Exchange factory for creating exchange instances."""
from typing import Dict, Any, Optional
from .base_exchange import BaseExchange
from .binance_exchange import BinanceExchange
import logging

logger = logging.getLogger(__name__)

class ExchangeFactory:
    """Factory class for creating exchange instances."""
    
    _exchanges = {
        "binance": BinanceExchange,
        # Add more exchanges here as they are implemented
        # "bybit": BybitExchange,
        # "kucoin": KucoinExchange,
    }
    
    @classmethod
    def create_exchange(cls, exchange_name: str, api_key: str, api_secret: str, 
                       testnet: bool = True, **kwargs) -> BaseExchange:
        """Create an exchange instance."""
        exchange_name = exchange_name.lower()
        
        if exchange_name not in cls._exchanges:
            available = ", ".join(cls._exchanges.keys())
            raise ValueError(f"Exchange {exchange_name} not supported. Available: {available}")
        
        exchange_class = cls._exchanges[exchange_name]
        return exchange_class(api_key, api_secret, testnet, **kwargs)
    
    @classmethod
    def get_supported_exchanges(cls) -> list:
        """Get list of supported exchanges."""
        return list(cls._exchanges.keys())
    
    @classmethod
    def register_exchange(cls, name: str, exchange_class: type):
        """Register a new exchange class."""
        if not issubclass(exchange_class, BaseExchange):
            raise ValueError("Exchange class must inherit from BaseExchange")
        
        cls._exchanges[name.lower()] = exchange_class
        logger.info(f"Registered exchange: {name}")
'
    
    # Configuration template
    safe_create_file "config/exchanges.yaml" '# Exchange Configuration
exchanges:
  binance:
    api_key: "your_binance_api_key_here"
    api_secret: "your_binance_api_secret_here"
    testnet: true
    
  # bybit:
  #   api_key: "your_bybit_api_key_here"
  #   api_secret: "your_bybit_api_secret_here"
  #   testnet: true

# Default exchange for trading
default_exchange: "binance"

# Trading pairs to monitor
trading_pairs:
  - "BTCUSDT"
  - "ETHUSDT"
  - "ADAUSDT"
  - "SOLUSDT"

# Rate limiting (requests per second)
rate_limits:
  binance: 10
  bybit: 5
'
}

add_strategies_feature() {
    log_info "Adding Trading Strategy Framework..."
    
    safe_create_dir "src/${PROJECT_NAME}/strategies"
    safe_create_dir "src/${PROJECT_NAME}/indicators"
    
    # Strategy base class
    safe_create_file "src/${PROJECT_NAME}/strategies/__init__.py" '"""Trading strategies package."""'
    
    safe_create_file "src/${PROJECT_NAME}/strategies/base_strategy.py" '"""Base strategy class with comprehensive framework."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Signal:
    """Trading signal class."""
    
    def __init__(self, symbol: str, action: str, price: float, timestamp: datetime,
                 confidence: float = 0.5, metadata: Dict = None):
        self.symbol = symbol
        self.action = action.upper()  # BUY, SELL, HOLD
        self.price = price
        self.timestamp = timestamp
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp between 0-1
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Signal({self.action} {self.symbol} @ {self.price}, confidence={self.confidence:.2f})"

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.positions = {}
        self.signals_history = []
        self.performance_metrics = {}
        
        # Risk management parameters
        self.max_position_size = config.get("max_position_size", 0.1)  # 10% of portfolio
        self.stop_loss_pct = config.get("stop_loss_pct", 0.02)  # 2% stop loss
        self.take_profit_pct = config.get("take_profit_pct", 0.06)  # 6% take profit
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators needed for the strategy."""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        required_columns = ["open", "high", "low", "close", "volume"]
        return all(col in data.columns for col in required_columns)
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              current_price: float) -> float:
        """Calculate position size based on risk management rules."""
        # Simple position sizing based on max position size
        max_investment = portfolio_value * self.max_position_size
        
        # Adjust based on signal confidence
        confidence_factor = signal.confidence
        adjusted_investment = max_investment * confidence_factor
        
        # Calculate position size
        position_size = adjusted_investment / current_price
        
        return position_size
    
    def should_exit_position(self, symbol: str, current_price: float) -> Tuple[bool, str]:
        """Check if position should be exited based on risk management."""
        if symbol not in self.positions:
            return False, ""
        
        position = self.positions[symbol]
        entry_price = position["avg_price"]
        quantity = position["quantity"]
        
        if quantity == 0:
            return False, ""
        
        # Calculate P&L percentage
        if quantity > 0:  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, "STOP_LOSS"
        
        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return True, "TAKE_PROFIT"
        
        return False, ""
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """Update position tracking."""
        if symbol not in self.positions:
            self.positions[symbol] = {"quantity": 0, "avg_price": 0, "unrealized_pnl": 0}
        
        current_pos = self.positions[symbol]
        
        if current_pos["quantity"] == 0:
            # New position
            self.positions[symbol] = {
                "quantity": quantity,
                "avg_price": price,
                "unrealized_pnl": 0
            }
        else:
            # Update existing position
            total_cost = (current_pos["quantity"] * current_pos["avg_price"]) + (quantity * price)
            total_quantity = current_pos["quantity"] + quantity
            
            if abs(total_quantity) < 1e-8:  # Position closed
                self.positions[symbol] = {"quantity": 0, "avg_price": 0, "unrealized_pnl": 0}
            else:
                self.positions[symbol] = {
                    "quantity": total_quantity,
                    "avg_price": total_cost / total_quantity if total_quantity != 0 else 0,
                    "unrealized_pnl": 0
                }
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Get current portfolio summary."""
        total_value = 0
        positions_summary = {}
        
        for symbol, position in self.positions.items():
            if position["quantity"] != 0 and symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = position["quantity"] * current_price
                unrealized_pnl = position_value - (position["quantity"] * position["avg_price"])
                
                positions_summary[symbol] = {
                    "quantity": position["quantity"],
                    "avg_price": position["avg_price"],
                    "current_price": current_price,
                    "position_value": position_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl / (position["quantity"] * position["avg_price"]) * 100
                }
                
                total_value += position_value
        
        return {
            "total_portfolio_value": total_value,
            "positions": positions_summary,
            "num_positions": len([p for p in positions_summary.values() if p["quantity"] != 0])
        }
    
    def add_signal_to_history(self, signal: Signal):
        """Add signal to history for analysis."""
        self.signals_history.append({
            "timestamp": signal.timestamp,
            "symbol": signal.symbol,
            "action": signal.action,
            "price": signal.price,
            "confidence": signal.confidence,
            "metadata": signal.metadata
        })
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and parameters."""
        return {
            "name": self.name,
            "config": self.config,
            "total_signals": len(self.signals_history),
            "active_positions": len([p for p in self.positions.values() if p["quantity"] != 0]),
            "risk_parameters": {
                "max_position_size": self.max_position_size,
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct
            }
        }
'
    
    # Technical Indicators
    safe_create_file "src/${PROJECT_NAME}/indicators/__init__.py" '"""Technical indicators package."""'
    
    safe_create_file "src/${PROJECT_NAME}/indicators/technical_indicators.py" '"""Common technical indicators for trading strategies."""
import pandas as pd
import numpy as np
from typing import Union, Tuple

def sma(data: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return data.rolling(window=window).mean()

def ema(data: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return data.ewm(span=window).mean()

def rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence)."""
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands."""
    sma_line = sma(data, window)
    std = data.rolling(window=window).std()
    
    upper_band = sma_line + (std * num_std)
    lower_band = sma_line - (std * num_std)
    
    return upper_band, sma_line, lower_band

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
               k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return k_percent, d_percent

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R."""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    obv_values = []
    obv_current = 0
    
    for i in range(len(close)):
        if i == 0:
            obv_values.append(volume.iloc[i])
            obv_current = volume.iloc[i]
        else:
            if close.iloc[i] > close.iloc[i-1]:
                obv_current += volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv_current -= volume.iloc[i]
            # If close is same, OBV stays same
            obv_values.append(obv_current)
    
    return pd.Series(obv_values, index=close.index)
'
    
    # Sample SMA Strategy
    safe_create_file "src/${PROJECT_NAME}/strategies/sma_crossover.py" '"""Simple Moving Average Crossover Strategy."""
import pandas as pd
from typing import List
from .base_strategy import BaseStrategy, Signal
from ..indicators.technical_indicators import sma
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SMACrossoverStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy.
    
    Generates buy signals when short MA crosses above long MA,
    and sell signals when short MA crosses below long MA.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.short_window = config.get("short_window", 10)
        self.long_window = config.get("long_window", 30)
        self.min_confidence = config.get("min_confidence", 0.6)
        
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA indicators."""
        data = data.copy()
        data["sma_short"] = sma(data["close"], self.short_window)
        data["sma_long"] = sma(data["close"], self.long_window)
        
        # Calculate signal strength based on MA separation
        data["ma_separation"] = (data["sma_short"] - data["sma_long"]) / data["close"]
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on SMA crossover."""
        if not self.validate_data(data):
            logger.error("Invalid data format for SMA strategy")
            return []
        
        if len(data) < self.long_window + 1:
            logger.warning(f"Insufficient data for SMA calculation. Need {self.long_window + 1}, got {len(data)}")
            return []
        
        # Calculate indicators
        data = self.calculate_indicators(data)
        signals = []
        
        # Generate signals
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Skip if indicators are NaN
            if pd.isna(current["sma_short"]) or pd.isna(current["sma_long"]):
                continue
            
            symbol = current.get("symbol", "UNKNOWN")
            timestamp = current.name if hasattr(current.name, "timestamp") else datetime.now()
            
            # Bullish crossover: short MA crosses above long MA
            if (previous["sma_short"] <= previous["sma_long"] and 
                current["sma_short"] > current["sma_long"]):
                
                confidence = self._calculate_signal_confidence(current, "BUY")
                
                if confidence >= self.min_confidence:
                    signal = Signal(
                        symbol=symbol,
                        action="BUY",
                        price=current["close"],
                        timestamp=timestamp,
                        confidence=confidence,
                        metadata={
                            "sma_short": current["sma_short"],
                            "sma_long": current["sma_long"],
                            "ma_separation": current["ma_separation"],
                            "volume": current["volume"]
                        }
                    )
                    signals.append(signal)
                    self.add_signal_to_history(signal)
            
            # Bearish crossover: short MA crosses below long MA
            elif (previous["sma_short"] >= previous["sma_long"] and 
                  current["sma_short"] < current["sma_long"]):
                
                confidence = self._calculate_signal_confidence(current, "SELL")
                
                if confidence >= self.min_confidence:
                    signal = Signal(
                        symbol=symbol,
                        action="SELL",
                        price=current["close"],
                        timestamp=timestamp,
                        confidence=confidence,
                        metadata={
                            "sma_short": current["sma_short"],
                            "sma_long": current["sma_long"],
                            "ma_separation": current["ma_separation"],
                            "volume": current["volume"]
                        }
                    )
                    signals.append(signal)
                    self.add_signal_to_history(signal)
        
        logger.info(f"Generated {len(signals)} signals for SMA crossover strategy")
        return signals
    
    def _calculate_signal_confidence(self, row: pd.Series, action: str) -> float:
        """Calculate signal confidence based on various factors."""
        base_confidence = 0.6
        
        # Factor 1: MA separation (higher separation = higher confidence)
        ma_separation_factor = min(0.3, abs(row["ma_separation"]) * 100)
        
        # Factor 2: Volume confirmation (higher volume = higher confidence)
        volume_factor = 0.1 if row["volume"] > row.get("volume_avg", row["volume"]) else 0
        
        # Factor 3: Trend strength (can be enhanced with additional indicators)
        trend_factor = 0.1
        
        confidence = base_confidence + ma_separation_factor + volume_factor + trend_factor
        return min(0.95, confidence)  # Cap at 95%
'
    
    # Strategy configuration
    safe_create_file "config/strategies.yaml" '# Trading Strategy Configurations

strategies:
  sma_crossover:
    short_window: 10
    long_window: 30
    min_confidence: 0.6
    max_position_size: 0.1
    stop_loss_pct: 0.02
    take_profit_pct: 0.06
    
  # rsi_strategy:
  #   rsi_period: 14
  #   oversold_threshold: 30
  #   overbought_threshold: 70
  #   min_confidence: 0.7
  
  # macd_strategy:
  #   fast_period: 12
  #   slow_period: 26
  #   signal_period: 9
  #   min_confidence: 0.65

# Default strategy
default_strategy: "sma_crossover"

# Risk management defaults
risk_management:
  max_portfolio_risk: 0.02  # 2% of portfolio per trade
  max_positions: 5
  correlation_threshold: 0.7  # Avoid highly correlated positions
'
}

# Continue with more features...
add_backtesting_feature() {
    log_info "Adding Backtesting Engine..."
    
    safe_create_file "src/${PROJECT_NAME}/backtesting.py" '"""Comprehensive backtesting engine for trading strategies."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from .strategies.base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.metrics = {}
        self.positions_history = []
        self.signals_history = []
    
    def add_trade(self, trade: Dict[str, Any]):
        """Add a completed trade."""
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.trades:
            return {"error": "No trades to analyze"}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        losing_trades = len(trades_df[trades_df["pnl"] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df["pnl"].sum()
        avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float("inf")
        
        # Equity curve metrics
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            returns = equity_df["equity"].pct_change().dropna()
            
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            max_drawdown = self._calculate_max_drawdown(equity_df["equity"])
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        self.metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
        
        return self.metrics
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        return drawdown.min()

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.1% commission per trade
        self.current_capital = initial_capital
        self.positions = {}
        self.result = BacktestResult()
    
    def run_backtest(self, strategy: BaseStrategy, data: pd.DataFrame, 
                    start_date: Optional[datetime] = None, 
                    end_date: Optional[datetime] = None) -> BacktestResult:
        """Run backtest for a strategy on historical data."""
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Filter data by date range if specified
        if start_date or end_date:
            data = self._filter_data_by_date(data, start_date, end_date)
        
        if data.empty:
            logger.error("No data available for backtesting")
            return self.result
        
        # Reset state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.result = BacktestResult()
        
        # Generate signals for entire dataset
        signals = strategy.generate_signals(data)
        
        if not signals:
            logger.warning("No signals generated by strategy")
            return self.result
        
        # Process signals chronologically
        signals.sort(key=lambda x: x.timestamp)
        
        for signal in signals:
            self._process_signal(signal, strategy, data)
        
        # Close any remaining positions at the end
        self._close_all_positions(data.iloc[-1]["close"], data.index[-1])
        
        # Calculate final metrics
        self.result.calculate_metrics()
        
        logger.info(f"Backtest completed. Total trades: {len(self.result.trades)}")
        return self.result
    
    def _filter_data_by_date(self, data: pd.DataFrame, start_date: Optional[datetime], 
                           end_date: Optional[datetime]) -> pd.DataFrame:
        """Filter data by date range."""
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        return data
    
    def _process_signal(self, signal: Signal, strategy: BaseStrategy, data: pd.DataFrame):
        """Process a trading signal."""
        symbol = signal.symbol
        action = signal.action
        price = signal.price
        timestamp = signal.timestamp
        
        # Calculate position size
        position_size = strategy.calculate_position_size(signal, self.current_capital, price)
        
        # Apply commission
        commission_cost = position_size * price * self.commission
        
        if action == "BUY":
            self._open_long_position(symbol, position_size, price, timestamp, commission_cost)
        elif action == "SELL":
            if symbol in self.positions and self.positions[symbol]["quantity"] > 0:
                self._close_position(symbol, price, timestamp, commission_cost)
            else:
                self._open_short_position(symbol, position_size, price, timestamp, commission_cost)
        
        # Record equity curve
        portfolio_value = self._calculate_portfolio_value(data, timestamp)
        self.result.equity_curve.append({
            "timestamp": timestamp,
            "equity": portfolio_value
        })
        
        # Record signal
        self.result.signals_history.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "price": price,
            "confidence": signal.confidence
        })
    
    def _open_long_position(self, symbol: str, quantity: float, price: float, 
                          timestamp: datetime, commission: float):
        """Open a long position."""
        cost = quantity * price + commission
        
        if cost > self.current_capital:
            logger.warning(f"Insufficient capital for {symbol} long position")
            return
        
        self.positions[symbol] = {
            "quantity": quantity,
            "entry_price": price,
            "entry_time": timestamp,
            "side": "LONG",
            "commission_paid": commission
        }
        
        self.current_capital -= cost
        logger.debug(f"Opened long position: {quantity} {symbol} @ {price}")
    
    def _open_short_position(self, symbol: str, quantity: float, price: float, 
                           timestamp: datetime, commission: float):
        """Open a short position."""
        # For simplicity, assume we can short without margin requirements
        self.positions[symbol] = {
            "quantity": -quantity,  # Negative for short
            "entry_price": price,
            "entry_time": timestamp,
            "side": "SHORT",
            "commission_paid": commission
        }
        
        # Add proceeds from short sale (minus commission)
        self.current_capital += (quantity * price - commission)
        logger.debug(f"Opened short position: {quantity} {symbol} @ {price}")
    
    def _close_position(self, symbol: str, price: float, timestamp: datetime, commission: float):
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        quantity = abs(position["quantity"])
        entry_price = position["entry_price"]
        side = position["side"]
        entry_commission = position["commission_paid"]
        
        # Calculate P&L
        if side == "LONG":
            pnl = (price - entry_price) * quantity - commission - entry_commission
            self.current_capital += quantity * price - commission
        else:  # SHORT
            pnl = (entry_price - price) * quantity - commission - entry_commission
            self.current_capital -= quantity * price + commission
        
        # Record trade
        trade = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": price,
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "pnl": pnl,
            "pnl_pct": pnl / (quantity * entry_price) * 100,
            "commission": commission + entry_commission,
            "duration": timestamp - position["entry_time"]
        }
        
        self.result.add_trade(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.debug(f"Closed {side} position: {symbol} P&L: {pnl:.2f}")
    
    def _close_all_positions(self, final_price: float, timestamp: datetime):
        """Close all remaining positions at the end of backtest."""
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, final_price, timestamp, 0)
    
    def _calculate_portfolio_value(self, data: pd.DataFrame, timestamp: datetime) -> float:
        """Calculate current portfolio value."""
        total_value = self.current_capital
        
        # Add unrealized P&L from open positions
        for symbol, position in self.positions.items():
            # For simplicity, use the last known price
            # In a real implementation, you would get the price at the specific timestamp
            current_price = data[data.index <= timestamp]["close"].iloc[-1] if len(data[data.index <= timestamp]) > 0 else position["entry_price"]
            
            quantity = position["quantity"]
            entry_price = position["entry_price"]
            
            if quantity > 0:  # Long position
                unrealized_pnl = (current_price - entry_price) * quantity
            else:  # Short position
                unrealized_pnl = (entry_price - current_price) * abs(quantity)
            
            total_value += unrealized_pnl
        
        return total_value
'
}

# ============================================================================
# MAIN EXECUTION FLOW
# ============================================================================

show_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║        🚀 Crypto Trading Feature Generator v1.0           ║
║              Add Professional Trading Features            ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

show_current_features() {
    echo -e "\n${CYAN}Current Trading Features:${NC}"
    local found_features=false
    
    for feature in "${!TRADING_FEATURES[@]}"; do
        if check_feature_exists "$feature"; then
            echo -e "  ✅ ${TRADING_FEATURES[$feature]}"
            found_features=true
        fi
    done
    
    if [ "$found_features" = false ]; then
        echo -e "  ${YELLOW}No trading features detected${NC}"
    fi
}

show_available_features() {
    echo -e "\n${CYAN}Available Trading Features:${NC}"
    local counter=1
    
    for feature in exchange_api strategies backtesting data_feeds notifications risk_management portfolio paper_trading live_trading monitoring; do
        local status=""
        if check_feature_exists "$feature"; then
            status="${GREEN}[INSTALLED]${NC}"
        fi
        
        echo -e "${counter}) ${TRADING_FEATURES[$feature]} $status"
        echo -e "   ${FEATURE_DESCRIPTIONS[$feature]}"
        ((counter++))
    done
}

select_features() {
    show_available_features
    
    echo -e "\n${YELLOW}Select features to add:${NC}"
    echo "• Enter numbers separated by commas (e.g., 1,2,5)"
    echo "• Enter 'all' to install all missing features"
    echo "• Enter 'q' to quit"
    
    read -p "Your choice: " selection
    
    case $selection in
        q|Q) exit 0 ;;
        all) install_all_features ;;
        *) install_selected_features "$selection" ;;
    esac
}

install_selected_features() {
    local selection="$1"
    IFS=',' read -ra FEATURES <<< "$selection"
    
    for feature_num in "${FEATURES[@]}"; do
        feature_num=$(echo "$feature_num" | tr -d ' ')
        
        case $feature_num in
            1) add_exchange_api_feature ;;
            2) add_strategies_feature ;;
            3) add_backtesting_feature ;;
            4) echo "Data feeds feature coming soon..." ;;
            5) echo "Notifications feature coming soon..." ;;
            6) echo "Risk management feature coming soon..." ;;
            7) echo "Portfolio feature coming soon..." ;;
            8) echo "Paper trading feature coming soon..." ;;
            9) echo "Live trading feature coming soon..." ;;
            10) echo "Monitoring feature coming soon..." ;;
            *) log_warning "Invalid selection: $feature_num" ;;
        esac
    done
    
    update_requirements
    show_completion_summary
}

install_all_features() {
    log_info "Installing all available features..."
    add_exchange_api_feature
    add_strategies_feature
    add_backtesting_feature
    # Add more as they're implemented
    
    update_requirements
    show_completion_summary
}

update_requirements() {
    log_info "Updating requirements.txt..."
    
    # Collect all dependencies
    local all_deps=""
    for feature in "${!TRADING_FEATURES[@]}"; do
        if check_feature_exists "$feature" && [[ -n "${FEATURE_DEPENDENCIES[$feature]:-}" ]]; then
            all_deps+=" ${FEATURE_DEPENDENCIES[$feature]}"
        fi
    done
    
    if [ -n "$all_deps" ]; then
        echo -e "\n# Trading Features Dependencies" >> requirements.txt
        echo "$all_deps" | tr ' ' '\n' | sort -u >> requirements.txt
        log_success "Updated requirements.txt"
    fi
}

show_completion_summary() {
    echo -e "\n${GREEN}🎉 Features Added Successfully!${NC}"
    echo -e "\n${CYAN}Next Steps:${NC}"
    echo "1. Install dependencies: ${YELLOW}pip install -r requirements.txt${NC}"
    echo "2. Configure exchanges: ${YELLOW}edit config/exchanges.yaml${NC}"
    echo "3. Configure strategies: ${YELLOW}edit config/strategies.yaml${NC}"
    echo "4. Test your setup with the examples in each module"
    echo -e "\n${PURPLE}💡 Tip: Run this script again anytime to add more features!${NC}"
}

main() {
    setup_feature_registry
    show_banner
    
    echo -e "Working in project: ${GREEN}$PROJECT_NAME${NC}"
    show_current_features
    
    echo -e "\n${CYAN}What would you like to do?${NC}"
    echo "1) Add new trading features"
    echo "2) Show feature status"
    echo "3) Exit"
    
    read -p "Choose (1-3): " -n 1 -r choice
    echo
    
    case $choice in
        1) select_features ;;
        2) show_current_features ;;
        3) log_info "Goodbye!" ;;
        *) log_error "Invalid choice" ;;
    esac
}

# Run the script
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi