"""
Comprehensive Evaluation System for CBF, CF, and Hybrid Filtering
Calculates all metrics and creates comparison visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import all recommender systems
from data_preparation import get_cleaned_data
from content_lsa_recommender import FinalOptimizedRecommender
from collaborative_recommender import CollaborativeRecommender
from hybrid_recommender import HybridRecommender
from lsa_ann_recommender import initialize_lsa_ann_recommender

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation system for all recommender approaches
    """
    
    def __init__(self):
        self.results = {}
        self.data = get_cleaned_data()
        self.books = self.data['books']
        self.ratings = self.data['ratings']
        self.book_tags = self.data['book_tags']
        self.tags = self.data['tags']
        
        # Initialize systems
        self.systems = {}
        self._initialize_systems()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _initialize_systems(self):
        """Initialize all recommender systems"""
        print(" Initializing Recommender Systems...")
        
        try:
            # Content-Based Filtering (LSA)
            print(" Initializing Content-Based Filtering (LSA)...")
            self.systems['CBF_LSA'] = FinalOptimizedRecommender(
                self.books, self.ratings, self.book_tags, self.tags
            )
            print(" CBF (LSA) initialized")
            
            # Collaborative Filtering
            print(" Initializing Collaborative Filtering...")
            self.systems['CF'] = CollaborativeRecommender()
            print(" CF initialized")
            
            # Hybrid Filtering
            print(" Initializing Hybrid Filtering...")
            self.systems['Hybrid'] = HybridRecommender()
            print(" Hybrid initialized")
            
            # LSA + ANN (Advanced)
            print(" Initializing LSA + ANN...")
            self.systems['LSA_ANN'] = initialize_lsa_ann_recommender(
                self.books, self.ratings, self.book_tags, self.tags,
                enable_tuning=False  # Skip tuning for evaluation
            )
            print(" LSA + ANN initialized")
            
        except Exception as e:
            print(f" Error initializing systems: {e}")
    
    def calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics (MSE, RMSE, MAE)"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }
    
    def calculate_ranking_metrics(self, y_true, y_pred, k=10):
        """Calculate ranking metrics (Precision@K, Recall@K, F1@K, Accuracy)"""
        # Convert to binary (rating >= 4 is positive)
        y_true_binary = (y_true >= 4).astype(int)
        y_pred_binary = (y_pred >= 4).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        
        return {
            f'Precision@{k}': precision,
            f'Recall@{k}': recall,
            f'F1@{k}': f1,
            f'Accuracy@{k}': accuracy
        }
    
    def evaluate_system(self, system_name, system, sample_size=1000):
        """Evaluate a single recommender system"""
        print(f"\n Evaluating {system_name}...")
        
        try:
            # Get sample of users for evaluation
            sample_users = self.ratings['user_id'].value_counts().head(sample_size).index
            
            y_true = []
            y_pred = []
            
            for user_id in sample_users[:100]:  # Limit to 100 users for speed
                try:
                    # Get user's actual ratings
                    user_ratings = self.ratings[self.ratings['user_id'] == user_id]
                    
                    if len(user_ratings) < 5:  # Skip users with too few ratings
                        continue
                    
                    # Split user's ratings for evaluation
                    train_ratings, test_ratings = train_test_split(
                        user_ratings, test_size=0.2, random_state=42
                    )
                    
                    # Get predictions for test books
                    for _, test_rating in test_ratings.iterrows():
                        book_id = test_rating['book_id']
                        true_rating = test_rating['rating']
                        
                        try:
                            # Get prediction based on system type
                            if system_name == 'CBF_LSA':
                                # For CBF, get similar books and predict based on similarity
                                similar_books = system.get_similar_books(book_id, n_recommendations=5)
                                if similar_books:
                                    # Simple prediction based on similar books' ratings
                                    similar_book_ids = [book['book_id'] for book in similar_books]
                                    similar_ratings = self.ratings[
                                        (self.ratings['book_id'].isin(similar_book_ids)) &
                                        (self.ratings['user_id'] == user_id)
                                    ]['rating']
                                    pred_rating = similar_ratings.mean() if len(similar_ratings) > 0 else 3.0
                                else:
                                    pred_rating = 3.0
                            
                            elif system_name == 'CF':
                                # For CF, get user-based recommendations
                                recommendations = system.recommend_for_user(user_id, n_recommendations=10)
                                if recommendations and book_id in [rec['book_id'] for rec in recommendations]:
                                    pred_rating = 4.0  # Assume high rating for recommended items
                                else:
                                    pred_rating = 3.0
                            
                            elif system_name == 'Hybrid':
                                # For Hybrid, combine CBF and CF
                                recommendations = system.recommend_for_user(user_id, n_recommendations=10)
                                if recommendations and book_id in [rec['book_id'] for rec in recommendations]:
                                    pred_rating = 4.0
                                else:
                                    pred_rating = 3.0
                            
                            elif system_name == 'LSA_ANN':
                                # For LSA + ANN, use the neural network prediction
                                recommendations = system.recommend_for_user(user_id, n_recommendations=10)
                                if recommendations and book_id in [rec['book_id'] for rec in recommendations]:
                                    pred_rating = 4.0
                                else:
                                    pred_rating = 3.0
                            
                            y_true.append(true_rating)
                            y_pred.append(pred_rating)
                            
                        except Exception as e:
                            # Skip problematic predictions
                            continue
                
                except Exception as e:
                    # Skip problematic users
                    continue
            
            if len(y_true) == 0:
                print(f" No valid predictions for {system_name}")
                return None
            
            # Calculate metrics
            regression_metrics = self.calculate_regression_metrics(y_true, y_pred)
            ranking_metrics = self.calculate_ranking_metrics(y_true, y_pred)
            
            # Combine all metrics
            all_metrics = {**regression_metrics, **ranking_metrics}
            all_metrics['Sample_Size'] = len(y_true)
            
            print(f" {system_name} evaluated with {len(y_true)} predictions")
            return all_metrics
            
        except Exception as e:
            print(f" Error evaluating {system_name}: {e}")
            return None
    
    def evaluate_all_systems(self):
        """Evaluate all recommender systems"""
        print("\n Starting Comprehensive Evaluation...")
        print("=" * 60)
        
        for system_name, system in self.systems.items():
            metrics = self.evaluate_system(system_name, system)
            if metrics:
                self.results[system_name] = metrics
        
        print(f"\n Evaluation complete for {len(self.results)} systems")
        return self.results
    
    def create_comparison_histograms(self):
        """Create comparison histograms for all metrics"""
        if not self.results:
            print("❌ No results to visualize. Run evaluate_all_systems() first.")
            return
        
        print("\n Creating Comparison Histograms...")
        
        # Prepare data for visualization
        metrics_data = []
        for system_name, metrics in self.results.items():
            for metric_name, value in metrics.items():
                if metric_name != 'Sample_Size':  # Skip sample size
                    metrics_data.append({
                        'System': system_name,
                        'Metric': metric_name,
                        'Value': value
                    })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create subplots for different metric groups
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Recommender System Evaluation', fontsize=16, fontweight='bold')
        
        # Regression Metrics
        regression_metrics = ['MSE', 'RMSE', 'MAE']
        regression_data = df_metrics[df_metrics['Metric'].isin(regression_metrics)]
        
        if not regression_data.empty:
            sns.barplot(data=regression_data, x='Metric', y='Value', hue='System', ax=axes[0,0])
            axes[0,0].set_title('Regression Metrics (Lower is Better)', fontweight='bold')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Ranking Metrics
        ranking_metrics = [col for col in df_metrics['Metric'].unique() if '@' in col]
        ranking_data = df_metrics[df_metrics['Metric'].isin(ranking_metrics)]
        
        if not ranking_data.empty:
            sns.barplot(data=ranking_data, x='Metric', y='Value', hue='System', ax=axes[0,1])
            axes[0,1].set_title('Ranking Metrics (Higher is Better)', fontweight='bold')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Overall Performance Comparison
        # Calculate overall score (normalized)
        overall_scores = {}
        for system_name, metrics in self.results.items():
            # Lower MSE/RMSE/MAE is better, higher precision/recall/f1/accuracy is better
            score = 0
            if 'MSE' in metrics:
                score += (1 / (1 + metrics['MSE']))  # Invert MSE
            if 'RMSE' in metrics:
                score += (1 / (1 + metrics['RMSE']))  # Invert RMSE
            if 'MAE' in metrics:
                score += (1 / (1 + metrics['MAE']))  # Invert MAE
            if 'Precision@10' in metrics:
                score += metrics['Precision@10']
            if 'Recall@10' in metrics:
                score += metrics['Recall@10']
            if 'F1@10' in metrics:
                score += metrics['F1@10']
            if 'Accuracy@10' in metrics:
                score += metrics['Accuracy@10']
            
            overall_scores[system_name] = score
        
        # Plot overall scores
        systems = list(overall_scores.keys())
        scores = list(overall_scores.values())
        
        bars = axes[1,0].bar(systems, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1,0].set_title('Overall Performance Score', fontweight='bold')
        axes[1,0].set_ylabel('Combined Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{score:.3f}', ha='center', va='bottom')
        
        # Sample Size Comparison
        sample_sizes = [metrics.get('Sample_Size', 0) for metrics in self.results.values()]
        systems = list(self.results.keys())
        
        bars = axes[1,1].bar(systems, sample_sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1,1].set_title('Evaluation Sample Sizes', fontweight='bold')
        axes[1,1].set_ylabel('Number of Predictions')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, size in zip(bars, sample_sizes):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + max(sample_sizes)*0.01,
                          f'{size}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('comprehensive_evaluation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Comparison histograms saved as 'comprehensive_evaluation_comparison.png'")
    
    def print_detailed_results(self):
        """Print detailed evaluation results"""
        if not self.results:
            print(" No results to display. Run evaluate_all_systems() first.")
            return
        
        print("\n DETAILED EVALUATION RESULTS")
        print("=" * 80)
        
        for system_name, metrics in self.results.items():
            print(f"\n {system_name}")
            print("-" * 40)
            
            # Regression metrics
            print(" Regression Metrics:")
            for metric in ['MSE', 'RMSE', 'MAE']:
                if metric in metrics:
                    print(f"  {metric}: {metrics[metric]:.4f}")
            
            # Ranking metrics
            print("\n Ranking Metrics:")
            for metric in ['Precision@10', 'Recall@10', 'F1@10', 'Accuracy@10']:
                if metric in metrics:
                    print(f"  {metric}: {metrics[metric]:.4f}")
            
            # Sample size
            if 'Sample_Size' in metrics:
                print(f"\n📈 Sample Size: {metrics['Sample_Size']}")
        
        # Summary table
        print("\n SUMMARY TABLE")
        print("=" * 80)
        
        # Create summary DataFrame
        summary_data = []
        for system_name, metrics in self.results.items():
            row = {'System': system_name}
            for metric in ['MSE', 'RMSE', 'MAE', 'Precision@10', 'Recall@10', 'F1@10', 'Accuracy@10']:
                row[metric] = metrics.get(metric, 0)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False, float_format='%.4f'))
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline"""
        print(" Starting Complete Evaluation Pipeline")
        print("=" * 60)
        
        # Evaluate all systems
        self.evaluate_all_systems()
        
        # Print detailed results
        self.print_detailed_results()
        
        # Create comparison visualizations
        self.create_comparison_histograms()
        
        print("\n Complete evaluation finished!")
        return self.results

def main():
    """Main function to run comprehensive evaluation"""
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_complete_evaluation()
    return results

if __name__ == "__main__":
    results = main()
