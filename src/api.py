"""
Simplified FastAPI backend for the backtesting engine (no database required)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import subprocess
import os
import sys
import json
import uuid

from backtesting_engine import SimpleBacktestingEngine, BacktestResults
from ml_layer import ml_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Simple Backtesting Engine API",
    description="API for running backtests without database dependencies",
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

# In-memory storage for backtest results
backtest_results = {}
backtest_runs = {}

# Pydantic models for API
class BacktestRequest(BaseModel):
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    parameters: Dict[str, Any] = {}

class BacktestResponse(BaseModel):
    id: str
    strategy: str
    symbol: str
    status: str
    created_at: datetime

class MetricResponse(BaseModel):
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

class TradeResponse(BaseModel):
    strategy: str
    symbol: str
    timestamp: str
    action: str
    price: float
    quantity: float
    commission: float

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Simple backtesting engine is running"}

# Debug endpoint
@app.get("/debug/backtest-runs")
async def get_backtest_runs():
    """Debug endpoint to see all backtest runs"""
    return {
        "total_runs": len(backtest_runs),
        "total_results": len(backtest_results),
        "runs": backtest_runs,
        "results": {k: {
            "strategy": v.strategy,
            "symbol": v.symbol,
            "error": v.error
        } for k, v in backtest_results.items()}
    }

# Data endpoints
@app.get("/data/symbols")
async def get_available_symbols():
    """Get list of available symbols"""
    return {"symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"]}

@app.get("/data/prices/{symbol}")
async def get_price_data(symbol: str, limit: int = 100):
    """Get price data for a symbol"""
    try:
        # Fetch recent real price data
        engine = SimpleBacktestingEngine()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        price_data = engine.fetch_real_data(symbol, start_date, end_date)
        
        # Convert to the format expected by frontend
        prices = []
        for data_point in price_data[-limit:]:  # Get the most recent data
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
        logger.error(f"Error getting price data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtesting endpoints
@app.post("/backtest/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run a backtest"""
    try:
        # Generate unique ID
        run_id = str(uuid.uuid4())
        
        # Create backtest run record
        backtest_runs[run_id] = {
            "id": run_id,
            "strategy": request.strategy,
            "symbol": request.symbol,
            "status": "RUNNING",
            "created_at": datetime.now(),
            "completed_at": None,
            "error_message": None
        }
        
        # Run backtest in background
        background_tasks.add_task(
            execute_backtest,
            run_id,
            request.strategy,
            request.symbol,
            request.start_date,
            request.end_date,
            request.initial_capital,
            request.parameters
        )
        
        return BacktestResponse(
            id=run_id,
            strategy=request.strategy,
            symbol=request.symbol,
            status="RUNNING",
            created_at=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/status/{run_id}")
async def get_backtest_status(run_id: str):
    """Get backtest status"""
    try:
        if run_id not in backtest_runs:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        
        run_info = backtest_runs[run_id]
        
        return {
            "id": run_info["id"],
            "strategy": run_info["strategy"],
            "symbol": run_info["symbol"],
            "status": run_info["status"],
            "created_at": run_info["created_at"],
            "completed_at": run_info["completed_at"],
            "error_message": run_info["error_message"]
        }
    
    except Exception as e:
        logger.error(f"Error getting backtest status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Results endpoints
@app.get("/results/metrics")
async def get_metrics(strategy: Optional[str] = None, symbol: Optional[str] = None):
    """Get performance metrics"""
    try:
        metrics = []
        
        for run_id, results in backtest_results.items():
            if results.error:
                continue
                
            # Apply filters
            if strategy and results.strategy != strategy:
                continue
            if symbol and results.symbol != symbol:
                continue
            
            metrics.append(MetricResponse(
                strategy=results.strategy,
                symbol=results.symbol,
                start_date=results.start_date,
                end_date=results.end_date,
                total_return=results.total_return,
                sharpe_ratio=results.sharpe_ratio,
                max_drawdown=results.max_drawdown,
                total_trades=results.total_trades,
                winning_trades=results.winning_trades,
                losing_trades=results.losing_trades,
                win_rate=results.win_rate,
                profit_factor=results.profit_factor
            ))
        
        return {"metrics": metrics}
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/trades")
async def get_trades(strategy: Optional[str] = None, symbol: Optional[str] = None, limit: int = 1000):
    """Get trade history"""
    try:
        trades = []
        
        for run_id, results in backtest_results.items():
            if results.error:
                continue
                
            # Apply filters
            if strategy and results.strategy != strategy:
                continue
            if symbol and results.symbol != symbol:
                continue
            
            for trade in results.trades[:limit]:
                trades.append(TradeResponse(
                    strategy=trade.strategy,
                    symbol=trade.symbol,
                    timestamp=trade.timestamp,
                    action=trade.action,
                    price=trade.price,
                    quantity=trade.quantity,
                    commission=trade.commission
                ))
        
        return {"trades": trades}
    
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ML endpoints
@app.post("/ml/train/{symbol}")
async def train_ml_models(symbol: str):
    """Train ML models for a symbol"""
    try:
        # Fetch real market data for training
        engine = SimpleBacktestingEngine()
        price_data = engine.fetch_real_data(symbol, "2023-01-01", "2024-12-31")
        
        # Convert to the format expected by ML manager
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
        
        # Train models
        results = ml_manager.train_models(symbol, ml_data)
        
        return {
            "message": f"ML models trained successfully for {symbol}",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error training ML models for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/signals/{symbol}")
async def get_ml_signals(symbol: str, limit: int = 100):
    """Get ML-generated signals"""
    try:
        # Generate recent price data for prediction
        engine = SimpleBacktestingEngine()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        price_data = engine.fetch_real_data(symbol, start_date, end_date)
        
        # Convert to the format expected by ML manager
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
        
        # Generate signals
        signals = ml_manager.generate_signals(symbol, ml_data)
        
        return {"signals": signals[:limit]}
        
    except Exception as e:
        logger.error(f"Error generating ML signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/status/{symbol}")
async def get_ml_status(symbol: str):
    """Get ML training status for a symbol"""
    try:
        status = ml_manager.get_training_status(symbol)
        return status
        
    except Exception as e:
        logger.error(f"Error getting ML status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def execute_backtest(run_id: str, strategy: str, symbol: str, 
                          start_date: str, end_date: str, initial_capital: float, 
                          parameters: Dict[str, Any]):
    """Execute backtest in background"""
    try:
        logger.info(f"Starting backtest {run_id}: {strategy} on {symbol}")
        
        # Update status to running
        if run_id in backtest_runs:
            backtest_runs[run_id]["status"] = "RUNNING"
        
        # Create engine and run backtest
        engine = SimpleBacktestingEngine()
        results = engine.run_backtest(
            strategy, symbol, start_date, end_date, initial_capital, parameters
        )
        
        logger.info(f"Backtest {run_id} completed. Error: {results.error}")
        logger.info(f"Results: total_return={results.total_return}, trades={results.total_trades}")
        
        # Store results
        backtest_results[run_id] = results
        
        # Update status
        if run_id in backtest_runs:
            if results.error:
                backtest_runs[run_id]["status"] = "FAILED"
                backtest_runs[run_id]["error_message"] = results.error
                logger.error(f"Backtest {run_id} failed: {results.error}")
            else:
                backtest_runs[run_id]["status"] = "COMPLETED"
                logger.info(f"Backtest {run_id} completed successfully")
            backtest_runs[run_id]["completed_at"] = datetime.now()
    
    except Exception as e:
        logger.error(f"Error executing backtest {run_id}: {e}")
        
        # Update status to failed
        if run_id in backtest_runs:
            backtest_runs[run_id]["status"] = "FAILED"
            backtest_runs[run_id]["error_message"] = str(e)
            backtest_runs[run_id]["completed_at"] = datetime.now()

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "src.api:app",
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', 8000)),
        reload=os.getenv('API_DEBUG', 'False').lower() == 'true'
    )
