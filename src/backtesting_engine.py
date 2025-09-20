"""
Simple Python backtesting engine with C++-style performance optimizations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import random
import yfinance as yf
import logging

@dataclass
class PriceData:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class Signal:
    action: str  # "BUY", "SELL", "HOLD"
    quantity: float
    confidence: float

@dataclass
class Trade:
    strategy: str
    symbol: str
    timestamp: str
    action: str
    price: float
    quantity: float
    commission: float

@dataclass
class BacktestResults:
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    trades: List[Trade]
    portfolio_values: List[Tuple[str, float]]
    error: str = ""

class Strategy:
    def __init__(self, params: Dict[str, float]):
        self.params = params
    
    def generate_signal(self, historical_data: List[PriceData], current_price: PriceData) -> Signal:
        raise NotImplementedError

class MovingAverageCrossover(Strategy):
    def __init__(self, params: Dict[str, float]):
        super().__init__(params)
        self.short_period = int(params.get("short_period", 10))
        self.long_period = int(params.get("long_period", 20))
    
    def generate_signal(self, historical_data: List[PriceData], current_price: PriceData) -> Signal:
        signal = Signal("HOLD", 0.0, 0.0)
        
        if len(historical_data) < self.long_period:
            return signal
        
        # Calculate moving averages using numpy for performance
        closes = np.array([p.close for p in historical_data])
        
        short_ma = np.mean(closes[-self.short_period:])
        long_ma = np.mean(closes[-self.long_period:])
        
        if len(historical_data) >= self.long_period + 1:
            prev_short_ma = np.mean(closes[-self.short_period-1:-1])
            prev_long_ma = np.mean(closes[-self.long_period-1:-1])
            
            # Bullish crossover
            if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                signal.action = "BUY"
                signal.quantity = 100.0
                signal.confidence = min(1.0, abs(short_ma - long_ma) / long_ma)
            # Bearish crossover
            elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                signal.action = "SELL"
                signal.quantity = 100.0
                signal.confidence = min(1.0, abs(short_ma - long_ma) / long_ma)
        
        return signal

class RSIStrategy(Strategy):
    def __init__(self, params: Dict[str, float]):
        super().__init__(params)
        self.period = int(params.get("period", 14))
        self.oversold = params.get("oversold", 30.0)
        self.overbought = params.get("overbought", 70.0)
    
    def calculate_rsi(self, data: List[PriceData], period: int) -> float:
        if len(data) < period + 1:
            return 50.0
        
        closes = np.array([p.close for p in data])
        deltas = np.diff(closes)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    
    def generate_signal(self, historical_data: List[PriceData], current_price: PriceData) -> Signal:
        signal = Signal("HOLD", 0.0, 0.0)
        
        if len(historical_data) < self.period + 1:
            return signal
        
        rsi = self.calculate_rsi(historical_data, self.period)
        
        if rsi < self.oversold:
            signal.action = "BUY"
            signal.quantity = 100.0
            signal.confidence = (self.oversold - rsi) / self.oversold
        elif rsi > self.overbought:
            signal.action = "SELL"
            signal.quantity = 100.0
            signal.confidence = (rsi - self.overbought) / (100.0 - self.overbought)
        
        return signal

class BollingerBandsStrategy(Strategy):
    def __init__(self, params: Dict[str, float]):
        super().__init__(params)
        self.period = int(params.get("period", 20))
        self.std_dev = params.get("std_dev", 2.0)
    
    def calculate_bollinger_bands(self, data: List[PriceData], period: int, std_dev: float) -> Tuple[float, float, float]:
        closes = np.array([p.close for p in data[-period:]])
        
        sma = np.mean(closes)
        std = np.std(closes)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    def generate_signal(self, historical_data: List[PriceData], current_price: PriceData) -> Signal:
        signal = Signal("HOLD", 0.0, 0.0)
        
        if len(historical_data) < self.period:
            return signal
        
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(
            historical_data, self.period, self.std_dev
        )
        
        if current_price.close <= lower_band:
            signal.action = "BUY"
            signal.quantity = 100.0
            signal.confidence = min(1.0, (lower_band - current_price.close) / lower_band)
        elif current_price.close >= upper_band:
            signal.action = "SELL"
            signal.quantity = 100.0
            signal.confidence = min(1.0, (current_price.close - upper_band) / upper_band)
        
        return signal

class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.trade_history: List[Trade] = []
    
    def execute_trade(self, trade: Trade) -> bool:
        total_cost = trade.price * trade.quantity + trade.commission
        
        if trade.action == "BUY":
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.positions[trade.symbol] = self.positions.get(trade.symbol, 0.0) + trade.quantity
                self.trade_history.append(trade)
                return True
        elif trade.action == "SELL":
            if self.positions.get(trade.symbol, 0.0) >= trade.quantity:
                self.cash += (trade.price * trade.quantity - trade.commission)
                self.positions[trade.symbol] -= trade.quantity
                self.trade_history.append(trade)
                return True
        
        return False
    
    def get_total_value(self, current_price: float) -> float:
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            total_value += quantity * current_price
        return total_value

class SimpleBacktestingEngine:
    """Simple backtesting engine with multiple strategies"""
    def __init__(self):
        self.strategies = {
            "moving_average_crossover": MovingAverageCrossover,
            "rsi_strategy": RSIStrategy,
            "bollinger_bands": BollingerBandsStrategy
        }
    
    def fetch_real_data(self, symbol: str, start_date: str, end_date: str) -> List[PriceData]:
        """Fetch real market data from Yahoo Finance"""
        try:
            logging.info(f"Fetching real data for {symbol} from {start_date} to {end_date}")
            
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logging.warning(f"No data found for {symbol}, falling back to sample data")
                return self.generate_sample_data(symbol, start_date, end_date)
            
            # Convert to our PriceData format
            price_data_list = []
            for timestamp, row in data.iterrows():
                price_data = PriceData(
                    timestamp=timestamp.strftime('%Y-%m-%d'),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                )
                price_data_list.append(price_data)
            
            logging.info(f"Successfully fetched {len(price_data_list)} data points for {symbol}")
            return price_data_list
            
        except Exception as e:
            logging.error(f"Error fetching real data for {symbol}: {e}")
            logging.info(f"Falling back to sample data for {symbol}")
            return self.generate_sample_data(symbol, start_date, end_date)
    
    def generate_sample_data(self, symbol: str, start_date: str, end_date: str) -> List[PriceData]:
        """Generate sample price data for testing (fallback when real data fails)"""
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        data = []
        current_date = start
        base_price = 100.0
        
        while current_date <= end:
            # Random walk with some trend
            change = random.uniform(-2, 2) + 0.1  # Slight upward bias
            base_price = max(10.0, base_price + change)
            
            price_data = PriceData(
                timestamp=current_date.isoformat(),
                open=base_price + random.uniform(-1, 1),
                high=base_price + random.uniform(0, 3),
                low=base_price - random.uniform(0, 3),
                close=base_price,
                volume=int(random.uniform(1000000, 5000000))
            )
            
            data.append(price_data)
            current_date += timedelta(days=1)
        
        return data
    
    def run_backtest(self, strategy_name: str, symbol: str, start_date: str, 
                    end_date: str, initial_capital: float, parameters: Dict[str, float]) -> BacktestResults:
        """Run a backtest with the specified parameters"""
        
        results = BacktestResults(
            strategy=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=initial_capital,
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            trades=[],
            portfolio_values=[]
        )
        
        try:
            # Create strategy
            if strategy_name not in self.strategies:
                results.error = f"Unknown strategy: {strategy_name}"
                return results
            
            strategy = self.strategies[strategy_name](parameters)
            
            # Fetch real market data
            price_data = self.fetch_real_data(symbol, start_date, end_date)
            
            if not price_data:
                results.error = "No price data available"
                return results
            
            # Initialize portfolio
            portfolio = Portfolio(initial_capital)
            
            results.start_date = price_data[0].timestamp
            results.end_date = price_data[-1].timestamp
            
            # Run simulation
            lookback = int(parameters.get("lookback", 50))
            
            for i in range(lookback, len(price_data)):
                current_price = price_data[i]
                
                # Get historical data for strategy
                historical_data = price_data[max(0, i - lookback):i + 1]
                
                # Get signal from strategy
                signal = strategy.generate_signal(historical_data, current_price)
                
                # Execute trade if signal is valid
                if signal.action != "HOLD":
                    trade = Trade(
                        strategy=strategy_name,
                        symbol=symbol,
                        timestamp=current_price.timestamp,
                        action=signal.action,
                        price=current_price.close,
                        quantity=signal.quantity,
                        commission=0.001 * current_price.close * signal.quantity  # 0.1% commission
                    )
                    
                    if portfolio.execute_trade(trade):
                        results.trades.append(trade)
                
                # Update portfolio value
                portfolio_value = portfolio.get_total_value(current_price.close)
                results.portfolio_values.append((current_price.timestamp, portfolio_value))
            
            # Calculate final metrics
            results.final_capital = portfolio.get_total_value(price_data[-1].close)
            results.total_return = (results.final_capital - initial_capital) / initial_capital
            
            # Calculate performance metrics
            self._calculate_metrics(results, price_data)
            
        except Exception as e:
            results.error = f"Error during backtest: {str(e)}"
        
        return results
    
    def _calculate_metrics(self, results: BacktestResults, price_data: List[PriceData]):
        """Calculate performance metrics"""
        if not results.portfolio_values:
            return
        
        # Calculate returns
        values = [pv[1] for pv in results.portfolio_values]
        returns = np.diff(values) / values[:-1]
        
        # Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            results.sharpe_ratio = (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252)
        
        # Max drawdown
        peak = results.initial_capital
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        results.max_drawdown = max_dd
        
        # Trade statistics
        results.total_trades = len(results.trades)
        
        # Simple P&L calculation
        winning_trades = 0
        losing_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        
        for i in range(0, len(results.trades), 2):
            if i + 1 < len(results.trades):
                buy_trade = results.trades[i]
                sell_trade = results.trades[i + 1]
                
                if buy_trade.action == "BUY" and sell_trade.action == "SELL":
                    pnl = (sell_trade.price - buy_trade.price) * buy_trade.quantity
                    if pnl > 0:
                        winning_trades += 1
                        total_profit += pnl
                    else:
                        losing_trades += 1
                        total_loss += abs(pnl)
        
        results.winning_trades = winning_trades
        results.losing_trades = losing_trades
        results.win_rate = winning_trades / results.total_trades if results.total_trades > 0 else 0.0
        results.profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Backtesting Engine')
    parser.add_argument('--strategy', required=True, help='Strategy name')
    parser.add_argument('--symbol', required=True, help='Symbol to backtest')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--parameters', default='{}', help='Strategy parameters as JSON')
    
    args = parser.parse_args()
    
    try:
        parameters = json.loads(args.parameters)
    except json.JSONDecodeError:
        parameters = {}
    
    engine = SimpleBacktestingEngine()
    results = engine.run_backtest(
        args.strategy, args.symbol, args.start_date, 
        args.end_date, args.initial_capital, parameters
    )
    
    if results.error:
        print(f"Error: {results.error}")
        return 1
    
    print("Backtest Results:")
    print(f"Strategy: {results.strategy}")
    print(f"Symbol: {results.symbol}")
    print(f"Period: {results.start_date} to {results.end_date}")
    print(f"Initial Capital: ${results.initial_capital:,.2f}")
    print(f"Final Capital: ${results.final_capital:,.2f}")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Total Trades: {results.total_trades}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    
    return 0

if __name__ == "__main__":
    exit(main())

