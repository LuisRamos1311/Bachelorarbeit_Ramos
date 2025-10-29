"""Evaluation metrics and comparison framework."""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelEvaluator:
    """Evaluate and compare prediction models."""
    
    def __init__(self):
        self.results = []
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Directional Accuracy (percentage of correct direction predictions)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
            
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
    
    def evaluate_model(self, model_name: str, feature_set: str, 
                      y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate a single model and store results.
        
        Args:
            model_name: Name of the model
            feature_set: Name of the feature set used
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation results
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        result = {
            'Model': model_name,
            'Feature_Set': feature_set,
            **metrics
        }
        
        self.results.append(result)
        return result
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get all results as a DataFrame.
        
        Returns:
            DataFrame with all evaluation results
        """
        return pd.DataFrame(self.results)
    
    def get_best_models(self, metric: str = 'RMSE') -> pd.DataFrame:
        """
        Get best performing models for each feature set.
        
        Args:
            metric: Metric to use for ranking
            
        Returns:
            DataFrame with best models
        """
        df = self.get_results_dataframe()
        if df.empty:
            return df
            
        # For metrics where lower is better
        if metric in ['MSE', 'RMSE', 'MAE', 'MAPE']:
            idx = df.groupby('Feature_Set')[metric].idxmin()
        else:  # For metrics where higher is better
            idx = df.groupby('Feature_Set')[metric].idxmax()
            
        return df.loc[idx]
    
    def compare_feature_sets(self, model_name: str) -> pd.DataFrame:
        """
        Compare different feature sets for a specific model.
        
        Args:
            model_name: Name of the model to compare
            
        Returns:
            DataFrame with comparison results
        """
        df = self.get_results_dataframe()
        if df.empty:
            return df
            
        return df[df['Model'] == model_name].sort_values('RMSE')
    
    def print_summary(self):
        """Print summary of evaluation results."""
        df = self.get_results_dataframe()
        
        if df.empty:
            print("No results to display.")
            return
            
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nTotal evaluations: {len(df)}")
        print(f"Models tested: {df['Model'].nunique()}")
        print(f"Feature sets tested: {df['Feature_Set'].nunique()}")
        
        print("\n" + "-"*80)
        print("BEST OVERALL PERFORMANCE (by RMSE)")
        print("-"*80)
        best_overall = df.loc[df['RMSE'].idxmin()]
        print(f"Model: {best_overall['Model']}")
        print(f"Feature Set: {best_overall['Feature_Set']}")
        print(f"RMSE: {best_overall['RMSE']:.6f}")
        print(f"MAE: {best_overall['MAE']:.6f}")
        print(f"MAPE: {best_overall['MAPE']:.2f}%")
        print(f"R²: {best_overall['R2']:.4f}")
        print(f"Directional Accuracy: {best_overall['Directional_Accuracy']:.2f}%")
        
        print("\n" + "-"*80)
        print("BEST MODEL FOR EACH FEATURE SET")
        print("-"*80)
        best_per_feature = self.get_best_models('RMSE')
        print(best_per_feature[['Feature_Set', 'Model', 'RMSE', 'MAE', 'R2']].to_string(index=False))
        
        print("\n" + "-"*80)
        print("AVERAGE PERFORMANCE BY MODEL")
        print("-"*80)
        avg_by_model = df.groupby('Model')[['RMSE', 'MAE', 'MAPE', 'R2']].mean()
        print(avg_by_model.to_string())
        
        print("\n" + "="*80)
