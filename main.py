"""
Main script for cryptocurrency prediction evaluation.

This script evaluates modern forecasting methods (LSTM, GRU, Simple RNN, ARIMA)
with different input variable combinations for cryptocurrency price prediction.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import CryptoDataLoader
from models.predictors import LSTMPredictor, GRUPredictor, SimpleRNNPredictor, ARIMAPredictor
from evaluation.metrics import ModelEvaluator
from utils.visualization import ResultVisualizer

import numpy as np


def main():
    """Main execution function."""
    
    print("="*80)
    print("CRYPTOCURRENCY PREDICTION EVALUATION")
    print("Evaluation of Modern Forecasting Methods with Different Input Variables")
    print("="*80)
    
    # Configuration
    SYMBOL = "BTC-USD"  # Bitcoin
    PERIOD = "2y"  # 2 years of data
    SEQ_LENGTH = 30  # 30 days lookback
    TRAIN_SIZE = 0.8
    
    print(f"\nConfiguration:")
    print(f"  Cryptocurrency: {SYMBOL}")
    print(f"  Data Period: {PERIOD}")
    print(f"  Sequence Length: {SEQ_LENGTH} days")
    print(f"  Train/Test Split: {TRAIN_SIZE:.0%}/{1-TRAIN_SIZE:.0%}")
    
    # Step 1: Load and prepare data
    print("\n" + "-"*80)
    print("STEP 1: Loading and Preparing Data")
    print("-"*80)
    
    loader = CryptoDataLoader(symbol=SYMBOL, period=PERIOD)
    print(f"Loading {SYMBOL} data...")
    data = loader.load_data()
    print(f"Data loaded: {len(data)} records")
    
    print("Adding technical indicators...")
    data_with_features = loader.add_technical_indicators(data)
    print(f"Features added. Shape: {data_with_features.shape}")
    
    # Get feature sets
    feature_sets = loader.get_feature_sets()
    print(f"\nFeature sets to evaluate: {len(feature_sets)}")
    for name, features in feature_sets.items():
        print(f"  - {name}: {len(features)} features")
    
    # Step 2: Initialize models
    print("\n" + "-"*80)
    print("STEP 2: Initializing Prediction Models")
    print("-"*80)
    
    models = {
        'LSTM': LSTMPredictor(units=50, dropout=0.2, epochs=30, batch_size=32),
        'GRU': GRUPredictor(units=50, dropout=0.2, epochs=30, batch_size=32),
        'SimpleRNN': SimpleRNNPredictor(units=50, dropout=0.2, epochs=30, batch_size=32),
        'ARIMA': ARIMAPredictor(order=(5, 1, 0))
    }
    
    print(f"Models initialized: {list(models.keys())}")
    
    # Step 3: Evaluation
    print("\n" + "-"*80)
    print("STEP 3: Training and Evaluating Models")
    print("-"*80)
    
    evaluator = ModelEvaluator()
    visualizer = ResultVisualizer()
    
    total_experiments = len(models) * len(feature_sets)
    experiment_count = 0
    
    print(f"Total experiments to run: {total_experiments}")
    print()
    
    # Evaluate each model with each feature set
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        for feature_set_name, features in feature_sets.items():
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}] {model_name} with {feature_set_name}...")
            
            try:
                # Prepare data for this feature set
                X_train, X_test, y_train, y_test, scaler = loader.prepare_sequences(
                    data_with_features,
                    features=features,
                    seq_length=SEQ_LENGTH,
                    train_size=TRAIN_SIZE
                )
                
                print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
                
                # Train model
                print(f"  Training {model_name}...")
                model.fit(X_train, y_train)
                
                # Make predictions
                print(f"  Making predictions...")
                y_pred = model.predict(X_test)
                
                # Evaluate
                result = evaluator.evaluate_model(
                    model_name=model_name,
                    feature_set=feature_set_name,
                    y_true=y_test,
                    y_pred=y_pred
                )
                
                print(f"  Results: RMSE={result['RMSE']:.6f}, MAE={result['MAE']:.6f}, "
                      f"R²={result['R2']:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                continue
    
    # Step 4: Results Analysis
    print("\n" + "-"*80)
    print("STEP 4: Analyzing Results")
    print("-"*80)
    
    evaluator.print_summary()
    
    # Step 5: Visualization
    print("\n" + "-"*80)
    print("STEP 5: Creating Visualizations")
    print("-"*80)
    
    results_df = evaluator.get_results_dataframe()
    
    if not results_df.empty:
        print("\nGenerating plots...")
        visualizer.create_summary_report(results_df, output_dir='plots')
        
        # Save results to CSV
        results_df.to_csv('evaluation_results.csv', index=False)
        print("\nResults saved to evaluation_results.csv")
    else:
        print("\nNo results to visualize.")
    
    # Final Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nTotal experiments completed: {len(results_df)}")
    
    if not results_df.empty:
        best = results_df.loc[results_df['RMSE'].idxmin()]
        print(f"\nBest Configuration:")
        print(f"  Model: {best['Model']}")
        print(f"  Feature Set: {best['Feature_Set']}")
        print(f"  RMSE: {best['RMSE']:.6f}")
        print(f"  MAE: {best['MAE']:.6f}")
        print(f"  MAPE: {best['MAPE']:.2f}%")
        print(f"  R²: {best['R2']:.4f}")
        print(f"  Directional Accuracy: {best['Directional_Accuracy']:.2f}%")
    
    print("\n" + "="*80)
    print("Thank you for using the Cryptocurrency Prediction Evaluation Framework!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
