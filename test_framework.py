"""
Quick test script to verify basic functionality.
This script tests data loading and basic model functionality without running full training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from data.data_loader import CryptoDataLoader
from models.predictors import LSTMPredictor, GRUPredictor, SimpleRNNPredictor
from evaluation.metrics import ModelEvaluator


def test_data_loader():
    """Test data loading and preprocessing."""
    print("Testing data loader...")
    
    loader = CryptoDataLoader(symbol="BTC-USD", period="1mo")
    data = loader.load_data()
    
    assert data is not None, "Data should not be None"
    assert len(data) > 0, "Data should have records"
    print(f"✓ Data loaded: {len(data)} records")
    
    # Test technical indicators
    data_with_features = loader.add_technical_indicators(data)
    assert 'MA_7' in data_with_features.columns, "Technical indicators should be added"
    print(f"✓ Technical indicators added: {data_with_features.shape}")
    
    # Test feature sets
    feature_sets = loader.get_feature_sets()
    assert len(feature_sets) > 0, "Feature sets should be defined"
    print(f"✓ Feature sets defined: {len(feature_sets)}")
    
    return loader, data_with_features


def test_sequence_preparation(loader, data):
    """Test sequence preparation for models."""
    print("\nTesting sequence preparation...")
    
    features = ['Close', 'Volume']
    X_train, X_test, y_train, y_test, scaler = loader.prepare_sequences(
        data, features=features, seq_length=10, train_size=0.8
    )
    
    assert X_train.shape[0] > 0, "Training data should have samples"
    assert X_test.shape[0] > 0, "Test data should have samples"
    assert X_train.shape[2] == len(features), "Features dimension should match"
    
    print(f"✓ Sequences prepared: Train={X_train.shape}, Test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def test_models(X_train, X_test, y_train, y_test):
    """Test model initialization and basic functionality."""
    print("\nTesting models...")
    
    # Create small test dataset
    X_train_small = X_train[:20]
    y_train_small = y_train[:20]
    X_test_small = X_test[:10]
    y_test_small = y_test[:10]
    
    # Test LSTM
    print("  Testing LSTM...")
    lstm = LSTMPredictor(units=10, epochs=2, batch_size=8)
    lstm.fit(X_train_small, y_train_small)
    pred_lstm = lstm.predict(X_test_small)
    assert len(pred_lstm) == len(y_test_small), "Predictions should match test size"
    print(f"  ✓ LSTM working: predictions shape={pred_lstm.shape}")
    
    # Test GRU
    print("  Testing GRU...")
    gru = GRUPredictor(units=10, epochs=2, batch_size=8)
    gru.fit(X_train_small, y_train_small)
    pred_gru = gru.predict(X_test_small)
    assert len(pred_gru) == len(y_test_small), "Predictions should match test size"
    print(f"  ✓ GRU working: predictions shape={pred_gru.shape}")
    
    # Test Simple RNN
    print("  Testing Simple RNN...")
    rnn = SimpleRNNPredictor(units=10, epochs=2, batch_size=8)
    rnn.fit(X_train_small, y_train_small)
    pred_rnn = rnn.predict(X_test_small)
    assert len(pred_rnn) == len(y_test_small), "Predictions should match test size"
    print(f"  ✓ Simple RNN working: predictions shape={pred_rnn.shape}")
    
    return y_test_small, pred_lstm


def test_evaluator(y_true, y_pred):
    """Test evaluation metrics."""
    print("\nTesting evaluator...")
    
    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(
        model_name="TestModel",
        feature_set="test_features",
        y_true=y_true,
        y_pred=y_pred
    )
    
    assert 'RMSE' in result, "RMSE should be calculated"
    assert 'MAE' in result, "MAE should be calculated"
    assert 'R2' in result, "R2 should be calculated"
    
    print(f"✓ Metrics calculated: RMSE={result['RMSE']:.4f}, MAE={result['MAE']:.4f}")
    
    # Test results DataFrame
    df = evaluator.get_results_dataframe()
    assert len(df) == 1, "Results should have one entry"
    print(f"✓ Results dataframe created: {df.shape}")


def main():
    """Run all tests."""
    print("="*60)
    print("RUNNING BASIC FUNCTIONALITY TESTS")
    print("="*60)
    
    try:
        # Test data loading
        loader, data = test_data_loader()
        
        # Test sequence preparation
        X_train, X_test, y_train, y_test = test_sequence_preparation(loader, data)
        
        # Test models
        y_true, y_pred = test_models(X_train, X_test, y_train, y_test)
        
        # Test evaluator
        test_evaluator(y_true, y_pred)
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nThe framework is working correctly!")
        print("You can now run: python main.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
