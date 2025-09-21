"""
Simple Vercel API handler for the AI Backtesting Engine
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def handler(request):
    """Main Vercel handler"""
    try:
        # Try to import and run the full app
        from fastapi import FastAPI, HTTPException, BackgroundTasks
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from typing import List, Optional, Dict, Any
        from datetime import datetime, timedelta
        import logging
        import uuid

        from backtesting_engine import SimpleBacktestingEngine, BacktestResults
        from ml_layer import ml_manager
        
        # Initialize FastAPI app
        app = FastAPI(
            title="AI Backtesting Engine API",
            description="API for running AI-powered backtests with real market data",
            version="1.0.0"
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # In-memory storage
        backtest_runs: Dict[str, Dict[str, Any]] = {}
        backtest_results: Dict[str, BacktestResults] = {}

        # Pydantic models
        class BacktestRequest(BaseModel):
            strategy: str
            symbol: str
            start_date: str
            end_date: str
            initial_capital: float
            parameters: Dict[str, float]

        # Health check endpoint
        @app.get("/")
        async def health_check():
            return {"status": "healthy", "message": "AI Backtesting Engine API is running"}

        @app.get("/health")
        async def health():
            return {"status": "healthy", "message": "AI Backtesting Engine API is running"}

        # Simple backtest endpoint
        @app.post("/backtest/run")
        async def run_backtest(request: BacktestRequest):
            try:
                run_id = str(uuid.uuid4())
                
                # Store backtest run info
                backtest_runs[run_id] = {
                    "id": run_id,
                    "strategy": request.strategy,
                    "symbol": request.symbol,
                    "status": "RUNNING",
                    "created_at": datetime.now().isoformat()
                }
                
                # Run backtest
                engine = SimpleBacktestingEngine()
                results = engine.run_backtest(
                    strategy_name=request.strategy,
                    symbol=request.symbol,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=request.initial_capital,
                    parameters=request.parameters
                )
                
                # Store results
                backtest_results[run_id] = results
                backtest_runs[run_id]["status"] = "COMPLETED"
                
                return {
                    "id": run_id,
                    "strategy": request.strategy,
                    "symbol": request.symbol,
                    "status": "COMPLETED",
                    "results": {
                        "total_return": results.total_return,
                        "total_trades": results.total_trades,
                        "win_rate": results.win_rate
                    }
                }
                
            except Exception as e:
                return {"error": str(e)}

        @app.get("/backtest/status/{run_id}")
        async def get_backtest_status(run_id: str):
            if run_id not in backtest_runs:
                raise HTTPException(status_code=404, detail="Backtest not found")
            return backtest_runs[run_id]

        @app.get("/results/metrics")
        async def get_results_metrics():
            metrics = []
            for run_id, result in backtest_results.items():
                if not result.error:
                    metrics.append({
                        "strategy": result.strategy,
                        "symbol": result.symbol,
                        "total_return": result.total_return,
                        "total_trades": result.total_trades,
                        "win_rate": result.win_rate
                    })
            return {"metrics": metrics}

        @app.get("/data/symbols")
        async def get_symbols():
            return {"symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX"]}

        @app.get("/data/prices/{symbol}")
        async def get_price_data(symbol: str, limit: int = 100):
            try:
                engine = SimpleBacktestingEngine()
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

                price_data = engine.fetch_real_data(symbol, start_date, end_date)
                
                prices = []
                for data_point in price_data[-limit:]:
                    prices.append({
                        "timestamp": data_point.timestamp,
                        "open": data_point.open,
                        "high": data_point.high,
                        "low": data_point.low,
                        "close": data_point.close,
                        "volume": data_point.volume
                    })
                
                return {"prices": prices}
                
            except Exception as e:
                return {"error": str(e)}

        # ML endpoints
        @app.post("/ml/train/{symbol}")
        async def train_ml_models(symbol: str):
            try:
                engine = SimpleBacktestingEngine()
                price_data = engine.fetch_real_data(symbol, "2023-01-01", "2024-12-31")
                
                ml_data = []
                for data_point in price_data:
                    ml_data.append({
                        "timestamp": data_point.timestamp,
                        "open": data_point.open,
                        "high": data_point.high,
                        "low": data_point.low,
                        "close": data_point.close,
                        "volume": data_point.volume
                    })
                
                results = ml_manager.train_models(symbol, ml_data)
                
                return {
                    "message": f"ML models trained successfully for {symbol}",
                    "results": results
                }
                
            except Exception as e:
                return {"error": str(e)}

        @app.get("/ml/signals/{symbol}")
        async def get_ml_signals(symbol: str):
            try:
                engine = SimpleBacktestingEngine()
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                
                price_data = engine.fetch_real_data(symbol, start_date, end_date)
                
                ml_data = []
                for data_point in price_data:
                    ml_data.append({
                        "timestamp": data_point.timestamp,
                        "open": data_point.open,
                        "high": data_point.high,
                        "low": data_point.low,
                        "close": data_point.close,
                        "volume": data_point.volume
                    })
                
                signals = ml_manager.generate_signals(symbol, ml_data)
                
                return {
                    "symbol": symbol,
                    "signals": signals
                }
                
            except Exception as e:
                return {"error": str(e)}

        return app
        
    except ImportError as e:
        # Fallback simple handler if dependencies fail
        def simple_handler(request):
            return {
                "status": "error",
                "message": f"Import error: {str(e)}",
                "suggestion": "Check dependencies in requirements.txt"
            }
        return simple_handler