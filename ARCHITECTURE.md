# System Architecture

## Overview

This framework evaluates cryptocurrency prediction models with different input variable combinations.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
├─────────────────────────────────────────────────────────────────┤
│  main.py          │  example.py       │  test_framework.py      │
│  (Full Eval)      │  (Quick Demo)     │  (Unit Tests)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Framework                           │
├──────────────────┬──────────────────┬──────────────────────────┤
│  Data Module     │  Models Module   │  Evaluation Module       │
│                  │                  │                          │
│ • CryptoData     │ • BasePredictor  │ • ModelEvaluator        │
│   Loader         │ • LSTM           │ • Metrics Calculator    │
│ • Technical      │ • GRU            │ • Results Aggregator    │
│   Indicators     │ • Simple RNN     │ • Best Model Selector   │
│ • Sequence       │ • ARIMA          │                          │
│   Preparation    │                  │                          │
│ • Feature Sets   │                  │                          │
└──────────────────┴──────────────────┴──────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Visualization Module                         │
├─────────────────────────────────────────────────────────────────┤
│ • Prediction Plots  • Comparison Charts  • Heatmaps            │
│ • Feature Analysis  • Performance Reports                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       External Services                         │
├─────────────────────────────────────────────────────────────────┤
│ • Yahoo Finance API (via yfinance)                             │
│ • TensorFlow/Keras (Deep Learning)                             │
│ • statsmodels (ARIMA)                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. Data Acquisition
   └─> Yahoo Finance API
       └─> Historical OHLCV data

2. Feature Engineering
   └─> Technical Indicators
       ├─> Moving Averages (MA, EMA)
       ├─> Momentum (MACD, RSI)
       ├─> Volatility (Bollinger Bands)
       └─> Price Changes

3. Data Preprocessing
   └─> Sequence Creation
       ├─> Sliding Window (seq_length)
       ├─> Normalization (MinMaxScaler)
       └─> Train/Test Split

4. Model Training
   └─> For each (Model × Feature Set):
       ├─> Train on training data
       └─> Validate on test data

5. Evaluation
   └─> Calculate Metrics
       ├─> RMSE, MAE, MAPE
       ├─> R², MSE
       └─> Directional Accuracy

6. Results
   └─> Output Generation
       ├─> CSV files
       ├─> Visualization plots
       └─> Console summaries
```

## Module Details

### Data Module (`src/data/data_loader.py`)

**Classes:**
- `CryptoDataLoader`: Main data loading and preprocessing class

**Key Methods:**
- `load_data()`: Fetch cryptocurrency data
- `add_technical_indicators()`: Calculate technical features
- `prepare_sequences()`: Create time series sequences
- `get_feature_sets()`: Define feature combinations

**Feature Sets:**
1. price_only: [Close]
2. ohlcv: [Open, High, Low, Close, Volume]
3. price_ma: [Close, MA_7, MA_14, MA_30]
4. technical_basic: [Close, MA_7, MA_14, RSI, Volume]
5. technical_full: [Extended technical indicators]
6. all_features: [All available features]

### Models Module (`src/models/predictors.py`)

**Base Class:**
- `BasePredictor`: Abstract interface for all models

**Implementations:**
- `LSTMPredictor`: Long Short-Term Memory network
- `GRUPredictor`: Gated Recurrent Unit network
- `SimpleRNNPredictor`: Basic recurrent network
- `ARIMAPredictor`: Statistical time series model

**Common Interface:**
- `fit(X_train, y_train)`: Train the model
- `predict(X)`: Make predictions
- `get_name()`: Return model name

### Evaluation Module (`src/evaluation/metrics.py`)

**Classes:**
- `ModelEvaluator`: Comprehensive evaluation framework

**Metrics:**
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error
- R²: Coefficient of Determination
- Directional Accuracy: Correct direction predictions

**Key Methods:**
- `calculate_metrics()`: Compute all metrics
- `evaluate_model()`: Evaluate single model
- `get_results_dataframe()`: Get all results
- `get_best_models()`: Find best performers
- `print_summary()`: Display results

### Visualization Module (`src/utils/visualization.py`)

**Classes:**
- `ResultVisualizer`: Create charts and plots

**Plot Types:**
- Prediction vs Actual
- Model Comparison (Bar Charts)
- Performance Heatmaps
- Feature Importance Analysis
- Summary Reports

## Execution Workflow

### 1. Test Workflow (test_framework.py)
```
Start
  ↓
Test Data Loading
  ↓
Test Feature Engineering
  ↓
Test Model Training (small dataset)
  ↓
Test Evaluation Metrics
  ↓
Report Success/Failure
```

### 2. Example Workflow (example.py)
```
Start
  ↓
Load 3 months of data
  ↓
Select 3 feature sets
  ↓
Train 2 models (LSTM, GRU)
  ↓
Evaluate (6 experiments)
  ↓
Generate comparison plot
  ↓
Save results to CSV
```

### 3. Full Evaluation (main.py)
```
Start
  ↓
Load 2 years of data
  ↓
Calculate all technical indicators
  ↓
For each of 4 models:
  ↓
  For each of 6 feature sets:
    ↓
    Prepare sequences
    ↓
    Train model
    ↓
    Make predictions
    ↓
    Calculate metrics
    ↓
  Next feature set
  ↓
Next model
  ↓
Aggregate results (24 experiments)
  ↓
Generate comprehensive visualizations
  ↓
Save all results
  ↓
Print summary
```

## Configuration Parameters

### Data Configuration
- `SYMBOL`: Cryptocurrency symbol (e.g., "BTC-USD")
- `PERIOD`: Data period (e.g., "2y")
- `SEQ_LENGTH`: Lookback window (default: 30)
- `TRAIN_SIZE`: Train/test split ratio (default: 0.8)

### Model Configuration
- `units`: Number of neurons (default: 50)
- `dropout`: Dropout rate (default: 0.2)
- `epochs`: Training iterations (default: 30-50)
- `batch_size`: Batch size (default: 32)

### ARIMA Configuration
- `order`: (p, d, q) parameters (default: (5, 1, 0))

## Dependencies

### Core Dependencies
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scikit-learn**: ML utilities and metrics

### Deep Learning
- **tensorflow**: Neural network framework
- **keras**: High-level neural network API

### Time Series
- **statsmodels**: ARIMA implementation
- **prophet**: Optional forecasting tool

### Data & Visualization
- **yfinance**: Yahoo Finance data
- **matplotlib**: Plotting library
- **seaborn**: Statistical visualization

## Performance Considerations

### Memory Usage
- Depends on sequence length and data size
- Typical: 1-2 GB RAM for 2 years of data

### Execution Time
- Test: ~30 seconds
- Example: ~5-10 minutes
- Full: ~30-60 minutes (CPU) / ~10-20 minutes (GPU)

### Optimization Opportunities
1. Use GPU for neural network training
2. Reduce sequence length
3. Decrease epochs for faster iteration
4. Use parallel processing for different models

## Extension Points

### Adding New Models
1. Inherit from `BasePredictor`
2. Implement `fit()` and `predict()`
3. Add to models dictionary in main script

### Adding New Features
1. Add calculation in `add_technical_indicators()`
2. Include in feature sets
3. Document the feature

### Adding New Metrics
1. Add calculation in `calculate_metrics()`
2. Update result dataframe structure
3. Add to visualization

### Custom Cryptocurrencies
- Simply change `SYMBOL` parameter
- Any ticker supported by Yahoo Finance

## Error Handling

### Data Errors
- Network failures: Retry mechanism
- Invalid symbols: Validation
- Missing data: Interpolation/forward-fill

### Training Errors
- NaN values: Data cleaning
- Memory issues: Batch size reduction
- Convergence: Early stopping

### Evaluation Errors
- Division by zero: Safe computation
- Empty results: Validation checks

## Testing Strategy

### Unit Tests (test_framework.py)
- Data loading functionality
- Feature engineering
- Model initialization
- Metrics calculation

### Integration Tests (example.py)
- End-to-end workflow
- Multiple models and features
- Results generation

### Validation
- Syntax checking
- Security scanning (CodeQL)
- Dependency vulnerability checks

## Security

### Addressed Vulnerabilities
- Updated TensorFlow to 2.12.1+
- Updated scikit-learn to 1.0.1+
- All dependencies scanned

### Best Practices
- No hardcoded credentials
- Input validation
- Error handling
- Warning filters (specific only)

## Documentation

### Available Documentation
- **README.md**: Overview and basic usage
- **DOKUMENTATION.md**: Comprehensive German docs
- **QUICKSTART.md**: Quick start guide
- **ARCHITECTURE.md**: This file

### Code Documentation
- Docstrings for all classes
- Method-level documentation
- Inline comments for complex logic
- Type hints where applicable

---

**Version:** 1.0  
**Last Updated:** October 2025  
**Author:** Luis Ramos  
**Institution:** Hochschule (University)  
**Purpose:** Bachelor's Thesis in Business Informatics
