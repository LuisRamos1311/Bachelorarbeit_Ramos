"""Visualization utilities for results and analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class ResultVisualizer:
    """Visualize evaluation results and predictions."""
    
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str, feature_set: str, 
                        save_path: str = None):
        """
        Plot true vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            feature_set: Name of the feature set
            save_path: Path to save the plot
        """
        plt.figure(figsize=(14, 6))
        
        plt.plot(y_true, label='Actual', linewidth=2, alpha=0.7)
        plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
        
        plt.title(f'{model_name} - {feature_set}\nPrediction vs Actual', fontsize=14)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Normalized Price', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.close()
        
    def plot_metrics_comparison(self, results_df: pd.DataFrame, 
                               metric: str = 'RMSE',
                               save_path: str = None):
        """
        Plot comparison of models across feature sets.
        
        Args:
            results_df: DataFrame with evaluation results
            metric: Metric to visualize
            save_path: Path to save the plot
        """
        plt.figure(figsize=(14, 8))
        
        pivot_data = results_df.pivot(index='Feature_Set', 
                                      columns='Model', 
                                      values=metric)
        
        pivot_data.plot(kind='bar', width=0.8)
        plt.title(f'Model Comparison by {metric}', fontsize=14)
        plt.xlabel('Feature Set', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.close()
        
    def plot_heatmap(self, results_df: pd.DataFrame,
                    metric: str = 'RMSE',
                    save_path: str = None):
        """
        Plot heatmap of model performance.
        
        Args:
            results_df: DataFrame with evaluation results
            metric: Metric to visualize
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        pivot_data = results_df.pivot(index='Model',
                                      columns='Feature_Set',
                                      values=metric)
        
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn_r',
                   cbar_kws={'label': metric})
        plt.title(f'Model Performance Heatmap - {metric}', fontsize=14)
        plt.xlabel('Feature Set', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.close()
        
    def plot_feature_importance(self, results_df: pd.DataFrame,
                               model_name: str,
                               save_path: str = None):
        """
        Plot feature set performance for a specific model.
        
        Args:
            results_df: DataFrame with evaluation results
            model_name: Model to analyze
            save_path: Path to save the plot
        """
        model_data = results_df[results_df['Model'] == model_name].copy()
        model_data = model_data.sort_values('RMSE')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            bars = ax.bar(range(len(model_data)), model_data[metric])
            ax.set_xticks(range(len(model_data)))
            ax.set_xticklabels(model_data['Feature_Set'], rotation=45, ha='right')
            ax.set_title(f'{metric} by Feature Set', fontsize=12)
            ax.set_ylabel(metric, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Color the best bar
            if metric in ['RMSE', 'MAE', 'MAPE']:
                best_idx = model_data[metric].idxmin()
            else:
                best_idx = model_data[metric].idxmax()
            
            for i, bar in enumerate(bars):
                if model_data.index[i] == best_idx:
                    bar.set_color('green')
                else:
                    bar.set_color('steelblue')
        
        plt.suptitle(f'{model_name} - Feature Set Performance', fontsize=14, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.close()
        
    def create_summary_report(self, results_df: pd.DataFrame,
                             output_dir: str = 'plots'):
        """
        Create a comprehensive visualization report.
        
        Args:
            results_df: DataFrame with evaluation results
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics comparison
        for metric in ['RMSE', 'MAE', 'R2']:
            self.plot_metrics_comparison(
                results_df, 
                metric=metric,
                save_path=f'{output_dir}/comparison_{metric}.png'
            )
        
        # Heatmap
        self.plot_heatmap(
            results_df,
            metric='RMSE',
            save_path=f'{output_dir}/heatmap_RMSE.png'
        )
        
        # Feature importance for each model
        for model in results_df['Model'].unique():
            self.plot_feature_importance(
                results_df,
                model_name=model,
                save_path=f'{output_dir}/features_{model}.png'
            )
        
        print(f"\nAll plots saved to {output_dir}/")
