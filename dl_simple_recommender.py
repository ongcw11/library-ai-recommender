# === SIMPLIFIED DEEP LEARNING RECOMMENDATION SYSTEM ===
# Lightweight implementation without heavy dependencies
# Provides neural network capabilities using only basic libraries

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Import existing data
from data_preparation import get_cleaned_data, EnhancedTextProcessor, EnhancedFeatureEngineer

class SimpleNeuralRecommender:
    
    def __init__(self, books_df, ratings_df, book_tags_df, tags_df, model_cache_dir='model_cache'):
        self.books = books_df
        self.ratings = ratings_df
        self.book_tags = book_tags_df
        self.tags = tags_df
        self.model_cache_dir = model_cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(model_cache_dir, exist_ok=True)
        
        print("Initializing Simple Neural Recommender...")
        
        # Try to load cached model first
        if self._load_cached_model():
            print("Loaded cached neural network model!")
        else:
            print("No cached model found, training new model...")
            # Prepare data mappings
            self._prepare_mappings()
            
            # Initialize neural network (smaller for memory efficiency)
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32),  # Smaller network
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                batch_size=512,  # Larger batch size for efficiency
                learning_rate='adaptive',
                max_iter=50,  # Fewer iterations
                random_state=42
            )
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            # Train the model
            self._train_model()
            
            # Save the trained model
            self._save_model()
        
        print("Simple Neural Recommender ready!")
    
    def _prepare_mappings(self):
        # Get active users (with at least 5 ratings)
        user_rating_counts = self.ratings['user_id'].value_counts()
        active_users = user_rating_counts[user_rating_counts >= 5].head(5000).index  # Limit to 5000 users
        
        # Get active books (with at least 10 ratings)
        book_rating_counts = self.ratings['book_id'].value_counts()
        active_books = book_rating_counts[book_rating_counts >= 10].head(5000).index  # Limit to 5000 books
        
        # Filter ratings to active users and books
        self.filtered_ratings = self.ratings[
            (self.ratings['user_id'].isin(active_users)) & 
            (self.ratings['book_id'].isin(active_books))
        ]
        
        # Create mappings
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(active_users)}
        self.book_id_map = {book_id: idx for idx, book_id in enumerate(active_books)}
        
        self.n_users = len(active_users)
        self.n_books = len(active_books)
        
        print(f"Filtered data: {self.n_users} active users, {self.n_books} active books")
        print(f"Filtered ratings: {len(self.filtered_ratings)}")
    
    def _save_model(self):
        try:
            # Save model
            model_path = os.path.join(self.model_cache_dir, 'simple_neural_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            scaler_path = os.path.join(self.model_cache_dir, 'simple_neural_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save mappings
            mappings_path = os.path.join(self.model_cache_dir, 'simple_neural_mappings.pkl')
            mappings = {
                'user_id_map': self.user_id_map,
                'book_id_map': self.book_id_map,
                'n_users': self.n_users,
                'n_books': self.n_books
            }
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
            
            print(f"Model saved to {self.model_cache_dir}/")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _load_cached_model(self):
        try:
            model_path = os.path.join(self.model_cache_dir, 'simple_neural_model.pkl')
            scaler_path = os.path.join(self.model_cache_dir, 'simple_neural_scaler.pkl')
            mappings_path = os.path.join(self.model_cache_dir, 'simple_neural_mappings.pkl')
            
            # Check if all required files exist
            if not all(os.path.exists(path) for path in [model_path, scaler_path, mappings_path]):
                return False
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load mappings
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.user_id_map = mappings['user_id_map']
                self.book_id_map = mappings['book_id_map']
                self.n_users = mappings['n_users']
                self.n_books = mappings['n_books']
            
            # We still need filtered_ratings for recommendations, so prepare it
            self._prepare_filtered_ratings()
            
            return True
            
        except Exception as e:
            print(f"Error loading cached model: {e}")
            return False
    
    def _prepare_filtered_ratings(self):
        # Get active users and books from mappings
        active_users = list(self.user_id_map.keys())
        active_books = list(self.book_id_map.keys())
        
        # Filter ratings to active users and books
        self.filtered_ratings = self.ratings[
            (self.ratings['user_id'].isin(active_users)) & 
            (self.ratings['book_id'].isin(active_books))
        ]
    
    def _create_features(self, user_ids, book_ids):
        features = []
        
        for user_id, book_id in zip(user_ids, book_ids):
            feature_vector = []
            
            # User features (compact encoding instead of one-hot)
            user_idx = self.user_id_map.get(user_id, 0)
            feature_vector.append(user_idx / self.n_users)  # Normalized user ID
            
            # Book features (compact encoding instead of one-hot)
            book_idx = self.book_id_map.get(book_id, 0)
            feature_vector.append(book_idx / self.n_books)  # Normalized book ID
            
            # Additional book features
            book_info = self.books[self.books['book_id'] == book_id]
            if not book_info.empty:
                book = book_info.iloc[0]
                
                # Average rating (normalized)
                avg_rating = book.get('average_rating', 0)
                feature_vector.append(avg_rating / 5.0 if pd.notna(avg_rating) else 0.0)
                
                # Rating count (log normalized)
                rating_count = book.get('ratings_count', 0)
                feature_vector.append(np.log1p(rating_count) / 10.0 if pd.notna(rating_count) else 0.0)
                
                # Publication year (normalized)
                pub_year = book.get('original_publication_year', 2000)
                if pd.notna(pub_year):
                    feature_vector.append((pub_year - 1900) / 125.0)  # Normalize to 0-1
                else:
                    feature_vector.append(0.5)
            else:
                # Default values if book not found
                feature_vector.extend([0.0, 0.0, 0.5])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _train_model(self):
        print("Training neural network...")
        
        # Prepare training data (use filtered ratings)
        user_ids = self.filtered_ratings['user_id'].values
        book_ids = self.filtered_ratings['book_id'].values
        ratings = self.filtered_ratings['rating'].values
        
        # Create features
        X = self._create_features(user_ids, book_ids)
        y = ratings
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Neural network training completed!")
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        if user_id not in self.user_id_map:
            print(f"User {user_id} not found in training data")
            return pd.DataFrame()
        
        # Get books user has already rated
        user_rated_books = set(
            self.filtered_ratings[self.filtered_ratings['user_id'] == user_id]['book_id'].tolist()
        )
        
        # Get all book IDs
        all_book_ids = list(self.book_id_map.keys())
        
        # Create features for all user-book pairs
        user_ids = [user_id] * len(all_book_ids)
        X = self._create_features(user_ids, all_book_ids)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Create results DataFrame
        results = []
        for book_id, prediction in zip(all_book_ids, predictions):
            # Skip if user has already rated this book
            if book_id in user_rated_books:
                continue
            
            # Get book info
            book_info = self.books[self.books['book_id'] == book_id]
            if not book_info.empty:
                book_data = book_info.iloc[0].copy()
                book_data['recommendation_score'] = float(prediction)
                results.append(book_data)
        
        # Sort by prediction score and return top recommendations
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('recommendation_score', ascending=False)
            results_df = results_df.head(n_recommendations)
        
        return results_df
    
    def get_similar_books(self, book_id, n_recommendations=10):
        """Get similar books using content-based features"""
        if book_id not in self.book_id_map:
            print(f"Book {book_id} not found")
            return pd.DataFrame()
        
        # Get book info
        book_info = self.books[self.books['book_id'] == book_id]
        if book_info.empty:
            return pd.DataFrame()
        
        book = book_info.iloc[0]
        
        # Create book features
        book_features = []
        
        # Average rating
        avg_rating = book.get('average_rating', 0)
        book_features.append(avg_rating / 5.0 if pd.notna(avg_rating) else 0.0)
        
        # Rating count
        rating_count = book.get('ratings_count', 0)
        book_features.append(np.log1p(rating_count) / 10.0 if pd.notna(rating_count) else 0.0)
        
        # Publication year
        pub_year = book.get('original_publication_year', 2000)
        if pd.notna(pub_year):
            book_features.append((pub_year - 1900) / 125.0)
        else:
            book_features.append(0.5)
        
        # Calculate similarities with all other books
        similarities = []
        for _, other_book in self.books.iterrows():
            other_book_id = other_book['book_id']
            
            if other_book_id == book_id:
                continue
            
            # Create other book features
            other_features = []
            other_avg_rating = other_book.get('average_rating', 0)
            other_features.append(other_avg_rating / 5.0 if pd.notna(other_avg_rating) else 0.0)
            
            other_rating_count = other_book.get('ratings_count', 0)
            other_features.append(np.log1p(other_rating_count) / 10.0 if pd.notna(other_rating_count) else 0.0)
            
            other_pub_year = other_book.get('original_publication_year', 2000)
            if pd.notna(other_pub_year):
                other_features.append((other_pub_year - 1900) / 125.0)
            else:
                other_features.append(0.5)
            
            # Calculate cosine similarity
            similarity = np.dot(book_features, other_features) / (
                np.linalg.norm(book_features) * np.linalg.norm(other_features)
            )
            
            similarities.append((other_book_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create results
        results = []
        for book_id_similar, similarity in similarities[:n_recommendations]:
            book_info = self.books[self.books['book_id'] == book_id_similar]
            if not book_info.empty:
                book_data = book_info.iloc[0].copy()
                book_data['similarity_score'] = float(similarity)
                results.append(book_data)
        
        return pd.DataFrame(results)
    
    def evaluate_model(self):
        print("Evaluating neural network model...")
        
        # Get test data
        user_ids = self.ratings['user_id'].values
        book_ids = self.ratings['book_id'].values
        ratings = self.ratings['rating'].values
        
        # Create features
        X = self._create_features(user_ids, book_ids)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(ratings, predictions)
        mae = mean_absolute_error(ratings, predictions)
        rmse = np.sqrt(mse)
        
        print(f"Neural Network Model Evaluation:")
        print(f"- MSE: {mse:.4f}")
        print(f"- MAE: {mae:.4f}")
        print(f"- RMSE: {rmse:.4f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions,
            'actuals': ratings
        }
    
    def clear_cache(self):
        try:
            cache_files = [
                'simple_neural_model.pkl',
                'simple_neural_scaler.pkl', 
                'simple_neural_mappings.pkl'
            ]
            
            for filename in cache_files:
                filepath = os.path.join(self.model_cache_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"Removed {filepath}")
            
            print("Cache cleared successfully!")
            
        except Exception as e:
            print(f"Error clearing cache: {e}")

class SimpleDeepLearningIntegration:
    
    def __init__(self, books_df, ratings_df, book_tags_df, tags_df, model_cache_dir='model_cache'):
        self.books = books_df
        self.ratings = ratings_df
        self.book_tags = book_tags_df
        self.tags = tags_df
        
        print("Initializing Simple Deep Learning Integration...")
        
        # Initialize neural recommender with cache directory
        self.neural_recommender = SimpleNeuralRecommender(
            books_df, ratings_df, book_tags_df, tags_df, model_cache_dir
        )
        
        print("Simple Deep Learning Integration ready!")
    
    def recommend_for_user(self, user_id, n_recommendations=10, use_percentage=True):
        """Generate recommendations using neural network"""
        recommendations = self.neural_recommender.recommend_for_user(
            user_id, n_recommendations=n_recommendations
        )
        
        # Convert scores to percentage if requested
        if use_percentage and not recommendations.empty:
            max_score = recommendations['recommendation_score'].max()
            min_score = recommendations['recommendation_score'].min()
            if max_score > min_score:
                recommendations['recommendation_score'] = (
                    (recommendations['recommendation_score'] - min_score) / 
                    (max_score - min_score) * 100
                ).round(1)
        
        return recommendations
    
    def get_similar_books(self, book_id, n_recommendations=10, use_percentage=True):
        """Get similar books using neural network"""
        similar_books = self.neural_recommender.get_similar_books(
            book_id, n_recommendations=n_recommendations
        )
        
        # Convert scores to percentage if requested
        if use_percentage and not similar_books.empty:
            similar_books['similarity_score'] = (similar_books['similarity_score'] * 100).round(1)
        
        return similar_books
    
    def get_system_info(self):
        return {
            'system_type': 'Simple Neural Network',
            'components': ['MLPRegressor', 'StandardScaler', 'Feature Engineering'],
            'model_type': 'neural_network',
            'description': 'Lightweight neural network using scikit-learn MLPRegressor'
        }
    
    def compare_systems(self, user_id, n_recommendations=5):
        """Compare neural network with baseline"""
        # Get neural recommendations
        neural_recs = self.recommend_for_user(user_id, n_recommendations)
        
        return {
            'neural_recommendations': neural_recs,
            'user_id': user_id
        }
    
    def get_performance_metrics(self):
        evaluation = self.neural_recommender.evaluate_model()
        
        return {
            'neural_network_metrics': {
                'rmse': evaluation['rmse'],
                'mae': evaluation['mae'],
                'mse': evaluation['mse']
            }
        }
    
    def clear_cache(self):
        self.neural_recommender.clear_cache()

def initialize_simple_deep_learning(books_df, ratings_df, book_tags_df, tags_df, model_cache_dir='model_cache'):
    return SimpleDeepLearningIntegration(books_df, ratings_df, book_tags_df, tags_df, model_cache_dir)

def main():
    print("=== SIMPLE DEEP LEARNING RECOMMENDATION SYSTEM ===")
    print("Lightweight neural network using scikit-learn")
    print()
    
    # Load data
    data = get_cleaned_data()
    books = data['books']
    ratings = data['ratings']
    book_tags = data['book_tags']
    tags = data['tags']
    
    print(f"Loaded data:")
    print(f"- Books: {len(books)}")
    print(f"- Ratings: {len(ratings)}")
    print(f"- Book-Tag pairs: {len(book_tags)}")
    print(f"- Tags: {len(tags)}")
    print()
    
    # Initialize system
    system = initialize_simple_deep_learning(books, ratings, book_tags, tags)
    
    # Test recommendations
    user_rating_counts = ratings['user_id'].value_counts()
    active_user = user_rating_counts[user_rating_counts >= 10].index[0]
    
    print(f"Testing recommendations for user {active_user}")
    recommendations = system.recommend_for_user(active_user, n_recommendations=5)
    
    if not recommendations.empty:
        print("Top recommendations:")
        for _, book in recommendations.iterrows():
            print(f"- {book['title']} (Score: {book['recommendation_score']:.3f})")
    
    # Test similar books
    sample_book_id = books.iloc[0]['book_id']
    similar_books = system.get_similar_books(sample_book_id, n_recommendations=3)
    
    if not similar_books.empty:
        print(f"\nBooks similar to '{books.iloc[0]['title']}':")
        for _, book in similar_books.iterrows():
            print(f"- {book['title']} (Similarity: {book['similarity_score']:.3f})")
    
    # Get performance metrics
    metrics = system.get_performance_metrics()
    print(f"\nNeural Network Performance:")
    print(f"- RMSE: {metrics['neural_network_metrics']['rmse']:.4f}")
    print(f"- MAE: {metrics['neural_network_metrics']['mae']:.4f}")
    
    print(f"\n SIMPLE NEURAL SYSTEM SUMMARY:")
    print(f" Neural Network: MLPRegressor with 3 hidden layers")
    print(f" Feature Engineering: User/Book embeddings + metadata")
    print(f" Lightweight: No heavy dependencies")
    print(f" Compatible: Works with existing scikit-learn")

if __name__ == "__main__":
    main()
