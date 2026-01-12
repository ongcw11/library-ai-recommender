"""
Data Visualization Module
Various visualization functions for the recommendation system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from data_preparation import get_cleaned_data

def plot_rating_distribution(ratings_df, save_path=None):
    """Plot the distribution of ratings"""
    plt.figure(figsize=(10, 6))
    
    # Rating distribution
    plt.subplot(1, 2, 1)
    ratings_df['rating'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Rating counts
    plt.subplot(1, 2, 2)
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    rating_counts.plot(kind='bar', color='lightcoral', alpha=0.7)
    plt.title('Rating Counts')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_user_activity(ratings_df, save_path=None):
    """Plot user activity patterns"""
    plt.figure(figsize=(15, 5))
    
    # User rating counts
    plt.subplot(1, 3, 1)
    user_counts = ratings_df['user_id'].value_counts()
    user_counts.hist(bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('User Rating Counts Distribution')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Number of Users')
    plt.grid(True, alpha=0.3)
    
    # Top users
    plt.subplot(1, 3, 2)
    top_users = user_counts.head(20)
    top_users.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Top 20 Most Active Users')
    plt.xlabel('User ID')
    plt.ylabel('Number of Ratings')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # User rating distribution
    plt.subplot(1, 3, 3)
    user_avg_ratings = ratings_df.groupby('user_id')['rating'].mean()
    user_avg_ratings.hist(bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Average Rating per User')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Users')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_book_popularity(ratings_df, books_df, save_path=None):
    """Plot book popularity patterns"""
    plt.figure(figsize=(15, 5))
    
    # Book rating counts
    plt.subplot(1, 3, 1)
    book_counts = ratings_df['book_id'].value_counts()
    book_counts.hist(bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title('Book Rating Counts Distribution')
    plt.xlabel('Number of Ratings per Book')
    plt.ylabel('Number of Books')
    plt.grid(True, alpha=0.3)
    
    # Top books
    plt.subplot(1, 3, 2)
    top_books = book_counts.head(20)
    top_books.plot(kind='bar', color='red', alpha=0.7)
    plt.title('Top 20 Most Rated Books')
    plt.xlabel('Book ID')
    plt.ylabel('Number of Ratings')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Book average ratings
    plt.subplot(1, 3, 3)
    book_avg_ratings = ratings_df.groupby('book_id')['rating'].mean()
    book_avg_ratings.hist(bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title('Average Rating per Book')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Books')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_publication_years(books_df, save_path=None):
    """Plot book publication year distribution"""
    plt.figure(figsize=(12, 6))
    
    # Filter valid years
    valid_years = books_df.dropna(subset=['original_publication_year'])
    valid_years = valid_years[valid_years['original_publication_year'] > 1800]
    valid_years = valid_years[valid_years['original_publication_year'] < 2025]
    
    # Publication year distribution
    plt.subplot(1, 2, 1)
    valid_years['original_publication_year'].hist(bins=50, alpha=0.7, color='gold', edgecolor='black')
    plt.title('Book Publication Year Distribution')
    plt.xlabel('Publication Year')
    plt.ylabel('Number of Books')
    plt.grid(True, alpha=0.3)
    
    # Decade distribution
    plt.subplot(1, 2, 2)
    decades = (valid_years['original_publication_year'] // 10) * 10
    decade_counts = decades.value_counts().sort_index()
    decade_counts.plot(kind='bar', color='coral', alpha=0.7)
    plt.title('Books by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Number of Books')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_language_distribution(books_df, save_path=None):
    """Plot book language distribution"""
    plt.figure(figsize=(12, 6))
    
    # Language distribution
    plt.subplot(1, 2, 1)
    language_counts = books_df['language_code'].value_counts().head(15)
    language_counts.plot(kind='bar', color='lightpink', alpha=0.7)
    plt.title('Top 15 Languages')
    plt.xlabel('Language Code')
    plt.ylabel('Number of Books')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Language pie chart
    plt.subplot(1, 2, 2)
    top_languages = books_df['language_code'].value_counts().head(10)
    other_count = books_df['language_code'].value_counts().iloc[10:].sum()
    if other_count > 0:
        top_languages['Other'] = other_count
    
    plt.pie(top_languages.values, labels=top_languages.index, autopct='%1.1f%%', startangle=90)
    plt.title('Language Distribution (Top 10)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_author_analysis(books_df, save_path=None):
    """Plot author analysis"""
    plt.figure(figsize=(15, 5))
    
    # Author book counts
    plt.subplot(1, 3, 1)
    author_counts = books_df['authors'].value_counts().head(20)
    author_counts.plot(kind='bar', color='lightsteelblue', alpha=0.7)
    plt.title('Top 20 Most Prolific Authors')
    plt.xlabel('Author')
    plt.ylabel('Number of Books')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Books per author distribution
    plt.subplot(1, 3, 2)
    books_per_author = books_df['authors'].value_counts()
    books_per_author.hist(bins=30, alpha=0.7, color='mediumpurple', edgecolor='black')
    plt.title('Books per Author Distribution')
    plt.xlabel('Number of Books')
    plt.ylabel('Number of Authors')
    plt.grid(True, alpha=0.3)
    
    # Author collaboration
    plt.subplot(1, 3, 3)
    # Count books with multiple authors
    multi_author = books_df['authors'].str.contains(',', na=False)
    single_author = ~multi_author
    collaboration_counts = pd.Series([single_author.sum(), multi_author.sum()], 
                                   index=['Single Author', 'Multiple Authors'])
    collaboration_counts.plot(kind='bar', color=['lightcoral', 'lightgreen'], alpha=0.7)
    plt.title('Author Collaboration')
    plt.xlabel('Author Type')
    plt.ylabel('Number of Books')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_recommendation_metrics(metrics_dict, save_path=None):
    """Plot recommendation system metrics"""
    plt.figure(figsize=(12, 8))
    
    # Extract metrics
    algorithms = list(metrics_dict.keys())
    rmse_scores = [metrics_dict[alg].get('rmse', 0) for alg in algorithms]
    mae_scores = [metrics_dict[alg].get('mae', 0) for alg in algorithms]
    precision_scores = [metrics_dict[alg].get('precision', 0) for alg in algorithms]
    recall_scores = [metrics_dict[alg].get('recall', 0) for alg in algorithms]
    
    # RMSE comparison
    plt.subplot(2, 2, 1)
    plt.bar(algorithms, rmse_scores, color='red', alpha=0.7)
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # MAE comparison
    plt.subplot(2, 2, 2)
    plt.bar(algorithms, mae_scores, color='blue', alpha=0.7)
    plt.title('MAE Comparison')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Precision comparison
    plt.subplot(2, 2, 3)
    plt.bar(algorithms, precision_scores, color='green', alpha=0.7)
    plt.title('Precision Comparison')
    plt.ylabel('Precision')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Recall comparison
    plt.subplot(2, 2, 4)
    plt.bar(algorithms, recall_scores, color='orange', alpha=0.7)
    plt.title('Recall Comparison')
    plt.ylabel('Recall')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_dashboard(books_df, ratings_df, save_path=None):
    """Create a comprehensive dashboard of all visualizations"""
    fig = plt.figure(figsize=(20, 15))
    
    # Rating distribution
    plt.subplot(3, 3, 1)
    ratings_df['rating'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    
    # User activity
    plt.subplot(3, 3, 2)
    user_counts = ratings_df['user_id'].value_counts()
    user_counts.hist(bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('User Activity')
    plt.xlabel('Ratings per User')
    plt.ylabel('Number of Users')
    
    # Book popularity
    plt.subplot(3, 3, 3)
    book_counts = ratings_df['book_id'].value_counts()
    book_counts.hist(bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Book Popularity')
    plt.xlabel('Ratings per Book')
    plt.ylabel('Number of Books')
    
    # Publication years
    plt.subplot(3, 3, 4)
    valid_years = books_df.dropna(subset=['original_publication_year'])
    valid_years = valid_years[valid_years['original_publication_year'] > 1800]
    valid_years['original_publication_year'].hist(bins=30, alpha=0.7, color='gold', edgecolor='black')
    plt.title('Publication Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Books')
    
    # Languages
    plt.subplot(3, 3, 5)
    language_counts = books_df['language_code'].value_counts().head(10)
    language_counts.plot(kind='bar', color='lightpink', alpha=0.7)
    plt.title('Top Languages')
    plt.xlabel('Language')
    plt.ylabel('Number of Books')
    plt.xticks(rotation=45)
    
    # Authors
    plt.subplot(3, 3, 6)
    author_counts = books_df['authors'].value_counts().head(10)
    author_counts.plot(kind='bar', color='lightsteelblue', alpha=0.7)
    plt.title('Top Authors')
    plt.xlabel('Author')
    plt.ylabel('Number of Books')
    plt.xticks(rotation=45)
    
    # Average ratings over time
    plt.subplot(3, 3, 7)
    if 'original_publication_year' in books_df.columns:
        book_ratings = ratings_df.merge(books_df[['book_id', 'original_publication_year']], on='book_id')
        yearly_ratings = book_ratings.groupby('original_publication_year')['rating'].mean()
        yearly_ratings.plot(color='purple', alpha=0.7)
        plt.title('Average Rating by Year')
        plt.xlabel('Publication Year')
        plt.ylabel('Average Rating')
    
    # Rating vs Popularity
    plt.subplot(3, 3, 8)
    book_stats = ratings_df.groupby('book_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    book_stats.columns = ['book_id', 'avg_rating', 'rating_count']
    plt.scatter(book_stats['rating_count'], book_stats['avg_rating'], alpha=0.5, color='red')
    plt.title('Rating vs Popularity')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Average Rating')
    
    # User rating patterns
    plt.subplot(3, 3, 9)
    user_avg_ratings = ratings_df.groupby('user_id')['rating'].mean()
    user_avg_ratings.hist(bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title('User Rating Patterns')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Users')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all visualizations"""
    print("📊 Creating Data Visualizations...")
    
    # Load data
    data = get_cleaned_data()
    books = data['books']
    ratings = data['ratings']
    
    print(f"Loaded {len(books)} books and {len(ratings)} ratings")
    
    # Create visualizations
    print("Creating rating distribution plot...")
    plot_rating_distribution(ratings)
    
    print("Creating user activity plot...")
    plot_user_activity(ratings)
    
    print("Creating book popularity plot...")
    plot_book_popularity(ratings, books)
    
    print("Creating publication years plot...")
    plot_publication_years(books)
    
    print("Creating language distribution plot...")
    plot_language_distribution(books)
    
    print("Creating author analysis plot...")
    plot_author_analysis(books)
    
    print("Creating comprehensive dashboard...")
    create_comprehensive_dashboard(books, ratings)
    
    print("✅ All visualizations completed!")

if __name__ == "__main__":
    main()
