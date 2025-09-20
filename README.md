# 🤖 AI Backtesting Engine

A lightweight, AI-powered backtesting engine built with Python, FastAPI, and scikit-learn. No database required - everything runs in memory for simplicity and speed.

## 🚀 Features

- **AI-Powered Backtesting**: Machine learning models for price prediction and trading signals
- **Multiple Strategies**: Moving Average Crossover, RSI, and Bollinger Bands
- **Real-time API**: FastAPI backend with comprehensive REST endpoints
- **Interactive Frontend**: Clean HTML dashboard with real-time updates
- **No Database**: In-memory storage for simplicity and fast performance

## 🧠 AI Capabilities

- **Random Forest Models**: Price prediction and direction classification
- **Feature Engineering**: 18+ technical indicators (MACD, RSI, Bollinger Bands, etc.)
- **Signal Generation**: BUY/SELL/HOLD recommendations with confidence scores

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup
```bash
git clone <repository-url>
cd backtesting-engine
uv sync
```

## 🚀 Quick Start

### 1. Start the API Server
```bash
uv run python main.py api
```
API available at `http://localhost:8000`

### 2. Start the Frontend Server
```bash
uv run python server.py
```
Dashboard available at `http://localhost:8080`

### 3. Open the Dashboard
Navigate to `http://localhost:8080` in your browser.

## 📊 Usage

1. **Select Strategy**: Choose from Moving Average Crossover, RSI, or Bollinger Bands
2. **Configure Parameters**: Set symbol, date range, initial capital, and strategy parameters
3. **Run Backtest**: Click "Run Backtest" and monitor progress
4. **View Results**: See performance metrics, trade history, and visualizations
5. **Train AI Models**: Click "Train ML Models" for AI-powered signals

## 🔧 API Endpoints

### Backtesting
- `POST /backtest/run` - Start a new backtest
- `GET /backtest/status/{run_id}` - Check backtest status
- `GET /results/metrics` - Get backtest results

### AI/ML
- `POST /ml/train/{symbol}` - Train ML models for a symbol
- `GET /ml/signals/{symbol}` - Get AI-generated signals

### Data
- `GET /data/symbols` - Get available symbols
- `GET /data/prices/{symbol}` - Get price data

## 🎯 Available Strategies

### Moving Average Crossover
- **Parameters**: Short period (default: 10), Long period (default: 20)
- **Logic**: Buy when short MA crosses above long MA, sell when it crosses below

### RSI Strategy
- **Parameters**: Period (default: 14), Oversold (default: 30), Overbought (default: 70)
- **Logic**: Buy when RSI < oversold, sell when RSI > overbought

### Bollinger Bands
- **Parameters**: Period (default: 20), Standard deviation (default: 2.0)
- **Logic**: Buy when price touches lower band, sell when price touches upper band

## 🤖 AI Model Details

### Feature Engineering
- **Moving Averages**: SMA (5, 10, 20, 50), EMA (12, 26)
- **MACD**: Line, signal, histogram
- **RSI**: 14-period relative strength index
- **Bollinger Bands**: Position, width, upper/lower bands
- **Price Changes**: 1-day, 5-day, 10-day returns
- **Volume**: Volume ratio, SMA
- **Volatility**: 5-day and 20-day rolling volatility

### Model Architecture
- **Random Forest Regressor**: Price prediction
- **Random Forest Classifier**: Direction prediction (up/down)
- **StandardScaler**: Feature normalization
- **Train/Test Split**: 80/20 split for model evaluation

## 📈 Example Usage

### Python API
```python
import requests

# Start a backtest
response = requests.post('http://localhost:8000/backtest/run', json={
    "strategy": "moving_average_crossover",
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000,
    "parameters": {
        "short_period": 10,
        "long_period": 20
    }
})

# Train AI models
requests.post('http://localhost:8000/ml/train/AAPL')

# Get AI signals
signals = requests.get('http://localhost:8000/ml/signals/AAPL')
```

### Command Line
```bash
# Test the backtesting engine
uv run python main.py test

# Start API server
uv run python main.py api
```

## 🔍 Debugging

- Click "Debug Info" in the frontend to see backtest runs and results
- Check the API logs for detailed execution information
- Use the `/debug/backtest-runs` endpoint for programmatic debugging

## 📋 Dependencies

- **FastAPI**: Web framework for the API
- **uvicorn**: ASGI server
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## 🎯 Performance

- **Backtests**: Complete in seconds for 1-year periods
- **ML Training**: ~2-3 seconds for 2 years of daily data
- **Memory Usage**: Minimal - all data stored in memory
- **API Response**: Sub-second response times for most endpoints

---

**Built with ❤️ using Python, FastAPI, and AI/ML technologies**