"""
Simplified main entry point for the backtesting engine (no database required)
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtesting_engine import SimpleBacktestingEngine
from api import app
import uvicorn

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('backtesting_engine.log')
        ]
    )

def run_api():
    """Run the FastAPI server"""
    print("Starting FastAPI server...")
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    debug = os.getenv('API_DEBUG', 'False').lower() == 'true'
    
    print(f"Server will be available at: http://{host}:{port}")
    print("API documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=debug
    )

def test_engine():
    """Test the backtesting engine directly"""
    print("Testing backtesting engine...")
    
    engine = SimpleBacktestingEngine()
    
    # Test with sample data
    results = engine.run_backtest(
        strategy_name="moving_average_crossover",
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
        parameters={
            "short_period": 10,
            "long_period": 20,
            "lookback": 50
        }
    )
    
    if results.error:
        print(f"❌ Error: {results.error}")
        return False
    
    print("✅ Backtest completed successfully!")
    print(f"Strategy: {results.strategy}")
    print(f"Symbol: {results.symbol}")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Total Trades: {results.total_trades}")
    
    return True

def main():
    """Main entry point for the backtesting engine"""
    parser = argparse.ArgumentParser(description='Simple Backtesting Engine')
    parser.add_argument('command', choices=[
        'test', 'api'
    ], help='Command to run')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    if args.command == 'test':
        success = test_engine()
        sys.exit(0 if success else 1)
    
    elif args.command == 'api':
        run_api()

if __name__ == "__main__":
    main()
