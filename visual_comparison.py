#!/usr/bin/env python3
"""
Visual comparison of LSA vs Deep Learning systems
Creates charts and visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_preparation import cleaned_after_outliers
from content_lsa_recommender import FinalOptimizedRecommender
import time

def create_performance_chart():
    """Create performance comparison chart"""
    
    # Performance data from our tests
    systems = ['LSA', 'Deep Learning']
    init_times = [23.976, 0.094]
    rec_times = [0.340, 4.208]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initialization time comparison
    bars1 = ax1.bar(systems, init_times, color=['#007bff', '#28a745'])
    ax1.set_title('System Initialization Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_yscale('log')  # Log scale due to large difference
    
    # Add value labels on bars
    for bar, time in zip(bars1, init_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # Recommendation time comparison
    bars2 = ax2.bar(systems, rec_times, color=['#007bff', '#28a745'])
    ax2.set_title('Recommendation Generation Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    
    # Add value labels on bars
    for bar, time in zip(bars2, rec_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Performance comparison chart saved as 'performance_comparison.png'")

def create_recommendation_analysis():
    """Create recommendation analysis visualization"""
    
    # Load data and test both systems
    books = cleaned_after_outliers['books']
    ratings = cleaned_after_outliers['ratings']
    book_tags = cleaned_after_outliers['book_tags']
    tags = cleaned_after_outliers['tags']
    
    # Initialize systems
    print("Initializing systems for analysis...")
    lsa_system = FinalOptimizedRecommender(books, ratings, book_tags, tags, n_components=100, precision_k=10)
    
    try:
        from dl_simple_recommender import initialize_simple_deep_learning
        dl_system = initialize_simple_deep_learning(books, ratings, book_tags, tags)
        dl_available = True
    except:
        dl_available = False
    
    # Test with multiple users
    user_ratings = ratings.groupby('user_id').size()
    test_users = user_ratings[user_ratings >= 20].index[:5]
    
    results = {
        'users': [],
        'lsa_times': [],
        'dl_times': [],
        'lsa_counts': [],
        'dl_counts': []
    }
    
    for user_id in test_users:
        print(f"Testing user {user_id}...")
        
        # LSA recommendations
        start_time = time.time()
        lsa_recs = lsa_system.recommend_for_user(user_id, n_recommendations=10, use_percentage=True)
        lsa_time = time.time() - start_time
        
        results['users'].append(f'User {user_id}')
        results['lsa_times'].append(lsa_time)
        results['lsa_counts'].append(len(lsa_recs))
        
        if dl_available:
            # Deep Learning recommendations
            start_time = time.time()
            dl_recs = dl_system.recommend_for_user(user_id, n_recommendations=10, use_percentage=True)
            dl_time = time.time() - start_time
            
            results['dl_times'].append(dl_time)
            results['dl_counts'].append(len(dl_recs))
        else:
            results['dl_times'].append(0)
            results['dl_counts'].append(0)
    
    # Create comparison charts
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    x = np.arange(len(results['users']))
    width = 0.35
    
    # Recommendation times
    ax1.bar(x - width/2, results['lsa_times'], width, label='LSA', color='#007bff')
    if dl_available:
        ax1.bar(x + width/2, results['dl_times'], width, label='Deep Learning', color='#28a745')
    ax1.set_title('Recommendation Time by User', fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results['users'])
    ax1.legend()
    
    # Recommendation counts
    ax2.bar(x - width/2, results['lsa_counts'], width, label='LSA', color='#007bff')
    if dl_available:
        ax2.bar(x + width/2, results['dl_counts'], width, label='Deep Learning', color='#28a745')
    ax2.set_title('Recommendation Count by User', fontweight='bold')
    ax2.set_ylabel('Number of Recommendations')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results['users'])
    ax2.legend()
    
    # Average performance
    avg_lsa_time = np.mean(results['lsa_times'])
    avg_dl_time = np.mean(results['dl_times']) if dl_available else 0
    avg_lsa_count = np.mean(results['lsa_counts'])
    avg_dl_count = np.mean(results['dl_counts']) if dl_available else 0
    
    categories = ['Avg Time (s)', 'Avg Count']
    lsa_values = [avg_lsa_time, avg_lsa_count]
    dl_values = [avg_dl_time, avg_dl_count] if dl_available else [0, 0]
    
    x3 = np.arange(len(categories))
    ax3.bar(x3 - width/2, lsa_values, width, label='LSA', color='#007bff')
    if dl_available:
        ax3.bar(x3 + width/2, dl_values, width, label='Deep Learning', color='#28a745')
    ax3.set_title('Average Performance Comparison', fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(categories)
    ax3.legend()
    
    # Performance ratio
    if dl_available and avg_lsa_time > 0:
        time_ratio = avg_dl_time / avg_lsa_time
        count_ratio = avg_dl_count / avg_lsa_count if avg_lsa_count > 0 else 0
        
        ratios = [time_ratio, count_ratio]
        ratio_labels = ['Time Ratio\n(DL/LSA)', 'Count Ratio\n(DL/LSA)']
        
        colors = ['red' if r > 1 else 'green' for r in ratios]
        ax4.bar(ratio_labels, ratios, color=colors)
        ax4.set_title('Performance Ratios', fontweight='bold')
        ax4.set_ylabel('Ratio (DL/LSA)')
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax4.set_ylim(0, max(ratios) * 1.2)
    
    plt.tight_layout()
    plt.savefig('recommendation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Recommendation analysis chart saved as 'recommendation_analysis.png'")

def main():
    """Main function to create visualizations"""
    print("🎨 Creating Visual Comparison Charts")
    print("=" * 40)
    
    # Create performance chart
    create_performance_chart()
    
    # Create recommendation analysis
    create_recommendation_analysis()
    
    print("\n✅ All visualizations created successfully!")
    print("📁 Files saved:")
    print("  • performance_comparison.png")
    print("  • recommendation_analysis.png")

if __name__ == "__main__":
    main()
