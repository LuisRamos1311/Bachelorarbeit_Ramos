"""
Example script demonstrating usage of the cryptocurrency prediction framework.
This runs a quick evaluation with reduced parameters for demonstration purposes.
"""

import sys
import os

# Add src to path before other imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from data.data_loader import CryptoDataLoader
from models.predictors import LSTMPredictor, GRUPredictor
from evaluation.metrics import ModelEvaluator
from utils.visualization import ResultVisualizer


def main():
    """Run example evaluation."""
    
    print("="*80)
    print("EXAMPLE: Cryptocurrency Prediction Evaluation")
    print("="*80)
    print("\nThis example demonstrates the framework with reduced parameters.")
    print("For full evaluation, run: python main.py\n")
    
    # Configuration - reduced for quick demonstration
    SYMBOL = "BTC-USD"
    PERIOD = "3mo"  # 3 months of data
    SEQ_LENGTH = 14  # 2 weeks lookback
    EPOCHS = 10  # Reduced epochs
    
    print(f"Configuration:")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Period: {PERIOD}")
    print(f"  Sequence Length: {SEQ_LENGTH} days")
    print(f"  Training Epochs: {EPOCHS}")
    
    # Load data
    print("\n" + "-"*80)
    print("Loading Data...")
    print("-"*80)
    
    loader = CryptoDataLoader(symbol=SYMBOL, period=PERIOD)
    data = loader.load_data()
    print(f"Loaded {len(data)} records")
    
    data_with_features = loader.add_technical_indicators(data)
    print(f"Added technical indicators")
    
    # Select a subset of feature sets for demo
    feature_sets = {
        'price_only': ['Close'],
        'ohlcv': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'technical_basic': ['Close', 'MA_7', 'MA_14', 'RSI', 'Volume']
    }
    
    # Initialize models - only LSTM and GRU for demo
    models = {
        'LSTM': LSTMPredictor(units=32, dropout=0.2, epochs=EPOCHS, batch_size=16),
        'GRU': GRUPredictor(units=32, dropout=0.2, epochs=EPOCHS, batch_size=16)
    }
    
    print(f"\nWill evaluate {len(models)} models with {len(feature_sets)} feature sets")
    print(f"Total experiments: {len(models) * len(feature_sets)}")
    
    # Evaluation
    print("\n" + "-"*80)
    print("Running Evaluations...")
    print("-"*80)
    
    evaluator = ModelEvaluator()
    experiment_count = 0
    total = len(models) * len(feature_sets)
    
    for model_name, model in models.items():
        for feature_set_name, features in feature_sets.items():
            experiment_count += 1
            print(f"\n[{experiment_count}/{total}] {model_name} + {feature_set_name}")
            
            try:
                # Prepare data
                X_train, X_test, y_train, y_test, scaler = loader.prepare_sequences(
                    data_with_features,
                    features=features,
                    seq_length=SEQ_LENGTH,
                    train_size=0.8
                )
                
                print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
                
                # Train
                print(f"  Training...")
                model.fit(X_train, y_train)
                
                # Predict
                print(f"  Predicting...")
                y_pred = model.predict(X_test)
                
                # Evaluate
                result = evaluator.evaluate_model(
                    model_name=model_name,
                    feature_set=feature_set_name,
                    y_true=y_test,
                    y_pred=y_pred
                )
                
                print(f"  RMSE: {result['RMSE']:.6f} | MAE: {result['MAE']:.6f} | R²: {result['R2']:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    results_df = evaluator.get_results_dataframe()
    
    if not results_df.empty:
        print("\nAll Results:")
        print(results_df[['Model', 'Feature_Set', 'RMSE', 'MAE', 'R2']].to_string(index=False))
        
        print("\n" + "-"*80)
        best = results_df.loc[results_df['RMSE'].idxmin()]
        print(f"\nBest Configuration:")
        print(f"  Model: {best['Model']}")
        print(f"  Feature Set: {best['Feature_Set']}")
        print(f"  RMSE: {best['RMSE']:.6f}")
        print(f"  MAE: {best['MAE']:.6f}")
        print(f"  R²: {best['R2']:.4f}")
        
        # Save results
        results_df.to_csv('example_results.csv', index=False)
        print(f"\nResults saved to example_results.csv")
        
        # Generate one comparison plot
        print("\nGenerating comparison plot...")
        visualizer = ResultVisualizer()
        visualizer.plot_metrics_comparison(
            results_df, 
            metric='RMSE',
            save_path='example_comparison.png'
        )
        
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)
    print("\nFor full evaluation with all models and feature sets, run:")
    print("  python main.py\n")


if __name__ == "__main__":
    main()
