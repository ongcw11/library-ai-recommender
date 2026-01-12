"""
Collaborative Filtering GUI Application
Desktop GUI for testing collaborative filtering recommendations
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from collaborative_recommender import CollaborativeRecommender
from data_preparation import get_cleaned_data

class CollaborativeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Collaborative Filtering Recommender System")
        self.root.geometry("800x600")
        
        # Initialize data (lazy loading)
        self._books = None
        self._ratings = None
        
        # Initialize recommender
        try:
            self.recommender = CollaborativeRecommender(n_components=50)
            self.recommender_ready = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize recommender: {e}")
            self.recommender_ready = False
        
        self.setup_ui()
    
    @property
    def books(self):
        if self._books is None:
            data = get_cleaned_data()
            self._books = data['books']
        return self._books
    
    @property
    def ratings(self):
        if self._ratings is None:
            data = get_cleaned_data()
            self._ratings = data['ratings']
        return self._ratings
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Collaborative Filtering Recommender", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # User ID input
        ttk.Label(main_frame, text="User ID:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.user_id_var = tk.StringVar()
        user_id_entry = ttk.Entry(main_frame, textvariable=self.user_id_var, width=20)
        user_id_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Number of recommendations
        ttk.Label(main_frame, text="Number of Recommendations:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.num_recs_var = tk.StringVar(value="10")
        num_recs_entry = ttk.Entry(main_frame, textvariable=self.num_recs_var, width=20)
        num_recs_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Algorithm selection
        ttk.Label(main_frame, text="Algorithm:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.algorithm_var = tk.StringVar(value="svd")
        algorithm_combo = ttk.Combobox(main_frame, textvariable=self.algorithm_var, 
                                     values=["svd", "knn"], state="readonly", width=17)
        algorithm_combo.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Get Recommendations", 
                  command=self.get_recommendations).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show User Stats", 
                  command=self.show_user_stats).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Recommendations", padding="10")
        results_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Treeview for results
        columns = ("Rank", "Book Title", "Author", "Predicted Rating", "Algorithm")
        self.tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def get_recommendations(self):
        """Get recommendations for the specified user"""
        if not self.recommender_ready:
            messagebox.showerror("Error", "Recommender not ready")
            return
        
        try:
            user_id = int(self.user_id_var.get())
            num_recs = int(self.num_recs_var.get())
            algorithm = self.algorithm_var.get()
            
            # Clear previous results
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Get recommendations
            recommendations = self.recommender.recommend_for_user(
                user_id, top_n=num_recs, mode=algorithm
            )
            
            if recommendations.empty:
                messagebox.showinfo("No Recommendations", 
                                  f"No recommendations found for user {user_id}")
                return
            
            # Display results
            for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
                book_info = self.books[self.books['book_id'] == rec['book_id']]
                if not book_info.empty:
                    book = book_info.iloc[0]
                    title = book.get('title', 'Unknown')[:50]
                    author = book.get('authors', 'Unknown')[:30]
                    rating = f"{rec.get('CF_score', 0):.3f}"
                    
                    self.tree.insert("", "end", values=(
                        i, title, author, rating, algorithm.upper()
                    ))
            
            messagebox.showinfo("Success", 
                              f"Found {len(recommendations)} recommendations for user {user_id}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for User ID and Number of Recommendations")
    except Exception as e:
            messagebox.showerror("Error", f"Failed to get recommendations: {e}")
    
    def clear_results(self):
        """Clear the results tree"""
        for item in self.tree.get_children():
            self.tree.delete(item)
    
    def show_user_stats(self):
        """Show statistics for the specified user"""
        try:
            user_id = int(self.user_id_var.get())
            
            # Get user ratings
            user_ratings = self.ratings[self.ratings['user_id'] == user_id]
            
            if user_ratings.empty:
                messagebox.showinfo("User Stats", f"User {user_id} has no ratings")
        return
            
            # Calculate statistics
            avg_rating = user_ratings['rating'].mean()
            rating_count = len(user_ratings)
            rating_std = user_ratings['rating'].std()
            
            # Get unique books
            unique_books = user_ratings['book_id'].nunique()
            
            stats_text = f"""User {user_id} Statistics:
            
Total Ratings: {rating_count}
Unique Books: {unique_books}
Average Rating: {avg_rating:.2f}
Rating Std Dev: {rating_std:.2f}
Min Rating: {user_ratings['rating'].min()}
Max Rating: {user_ratings['rating'].max()}"""
            
            messagebox.showinfo("User Statistics", stats_text)
            
    except ValueError:
            messagebox.showerror("Error", "Please enter a valid User ID")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get user stats: {e}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = CollaborativeGUI(root)
root.mainloop()

if __name__ == "__main__":
    main()
