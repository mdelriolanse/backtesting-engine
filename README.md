# 📈 Backtesting Engine

A comprehensive backtesting engine with machine learning capabilities, built with FastAPI, PostgreSQL, and Streamlit.

## 🚀 Features

- **Backtesting Engine**: Run backtests with multiple trading strategies
- **Machine Learning**: ML models for return and direction prediction
- **Real-time API**: FastAPI backend with comprehensive endpoints
- **Interactive Frontend**: Streamlit dashboard for easy interaction
- **Database**: PostgreSQL for data persistence
- **Sample Data**: Realistic stock data for testing

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL
- uv (Python package manager)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd backtesting-engine
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up PostgreSQL**
   - Install PostgreSQL on your system
   - Create a database named `backtesting_db`
   - Update `.env` file with your database credentials

4. **Initialize the system**
   ```bash
   uv run -- python main.py full-setup
   ```

## 🎯 Quick Start

### 1. Start the API Server
```bash
uv run -- python main.py api
```
The API will be available at: http://localhost:8000

### 2. Start the Frontend
```bash
uv run -- python run_frontend.py
```
The frontend will be available at: http://localhost:8501

### 3. Access the Dashboard
Open your browser and go to: http://localhost:8501

## 📊 Available Strategies

### Moving Average Crossover
- **Description**: Buy when short MA crosses above long MA, sell when it crosses below
- **Parameters**: 
  - `short_period`: Short moving average period (default: 10)
  - `long_period`: Long moving average period (default: 20)
  - `lookback`: Historical data window (default: 50)

### RSI Strategy
- **Description**: Buy when RSI is oversold, sell when overbought
- **Parameters**:
  - `period`: RSI calculation period (default: 14)
  - `oversold`: Oversold threshold (default: 30)
  - `overbought`: Overbought threshold (default: 70)

### Bollinger Bands
- **Description**: Buy when price touches lower band, sell when it touches upper band
- **Parameters**:
  - `period`: Moving average period (default: 20)
  - `std_dev`: Standard deviation multiplier (default: 2.0)
  - `lookback`: Historical data window (default: 50)

## 🔧 API Endpoints

### Backtesting
- `POST /backtest/run` - Run a backtest
- `GET /backtest/status/{run_id}` - Get backtest status

### Results
- `GET /results/metrics` - Get performance metrics
- `GET /results/trades` - Get trade history

### Machine Learning
- `POST /ml/train/{symbol}` - Train ML models
- `GET /ml/signals/{symbol}` - Get ML signals
- `POST /ml/generate/{symbol}` - Generate new ML signal

### Data
- `GET /data/symbols` - Get available symbols
- `GET /data/prices/{symbol}` - Get price data
- `POST /data/upload` - Upload price data

## 📈 Frontend Features

### Dashboard
- Overview of system status
- Recent performance metrics
- Quick statistics

### Run Backtest
- Interactive strategy selection
- Parameter tuning with sliders
- Real-time progress tracking
- Results visualization

### View Results
- Performance metrics table
- Interactive charts (Return vs Drawdown)
- Trade history
- Filtering by strategy and symbol

### ML Models
- Train models for specific symbols
- View ML-generated signals
- Signal confidence visualization
- Historical predictions

### Price Data
- Interactive candlestick charts
- Price statistics
- Historical data table

## 🗄️ Database Schema

### Tables
- **prices**: Historical OHLCV data
- **trades**: Individual trade records
- **metrics**: Strategy performance metrics
- **signals**: ML-generated trading signals
- **backtest_runs**: Backtest execution metadata

## 📁 Project Structure

```
backtesting-engine/
├── src/                    # Source code
│   ├── api.py             # FastAPI endpoints
│   ├── database.py        # Database connection
│   ├── models.py          # SQLAlchemy models
│   ├── ml_layer.py        # Machine learning
│   ├── data_ingestion.py  # Data loading
│   ├── backtesting_engine.cpp  # C++ engine
│   ├── strategy.h         # Strategy definitions
│   └── portfolio.h        # Portfolio management
├── examples/              # Sample data
│   └── data/             # CSV files
├── database/             # Database schema
├── frontend.py           # Streamlit frontend
├── main.py              # Main entry point
└── requirements.txt     # Dependencies
```

## 🔧 Configuration

### Environment Variables (.env)
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=backtesting_db
DB_USER=postgres
DB_PASSWORD=your_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# C++ Engine Configuration
CPP_ENGINE_PATH=./build/backtesting_engine
```

## 🚀 Commands

### Main Commands
```bash
# Initialize database
uv run -- python main.py init

# Load sample data
uv run -- python main.py load-data

# Train ML models
uv run -- python main.py train-ml

# Start API server
uv run -- python main.py api

# Full setup (init + load-data + train-ml)
uv run -- python main.py full-setup
```

### Frontend
```bash
# Start Streamlit frontend
uv run -- python run_frontend.py
```

## 📊 Sample Data

The system includes realistic sample data for:
- **AAPL** (Apple Inc.)
- **GOOGL** (Alphabet Inc.)
- **MSFT** (Microsoft Corporation)
- **TSLA** (Tesla Inc.)

Each symbol has 2 years of daily OHLCV data with realistic price movements and volume patterns.

## 🤖 Machine Learning

### Models
- **Return Prediction**: Random Forest regressor for predicting next-day returns
- **Direction Prediction**: Random Forest classifier for predicting price direction

### Features
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Lag features (1, 2, 3, 5 days)
- Volume indicators
- Price momentum

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Ensure PostgreSQL is running
   - Check database credentials in `.env`
   - Verify database `backtesting_db` exists

2. **API Server Not Starting**
   - Check if port 8000 is available
   - Verify all dependencies are installed
   - Check logs for specific errors

3. **Frontend Not Loading**
   - Ensure API server is running on port 8000
   - Check if port 8501 is available
   - Verify Streamlit is installed

4. **No Data Available**
   - Run `uv run -- python main.py load-data`
   - Check if sample data files exist in `examples/data/`

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at http://localhost:8000/docs
3. Open an issue on GitHub
