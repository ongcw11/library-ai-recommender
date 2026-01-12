"""
Enhanced Data Preparation Pipeline
Complete implementation with all best practices for recommender system data preparation
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Import data from the data loader module
from data_loader import books, ratings, book_tags, tags, to_read

# ========================================== Data Understanding ==========================================

def mem_mb(df):
    return df.memory_usage(deep=True).sum() / (1024**2)

def summarize_df(name, df):
    if df is None:
        print(f"\n{name}: [NOT LOADED]")
        return
    print(f"\n=== {name.upper()} ===")
    print(f"Shape: {df.shape} | Memory: {mem_mb(df):.2f} MB")
    print("Columns:", list(df.columns))

    # nulls & duplicates
    nulls = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values (top 10):")
    print(nulls.head(10))

    dups = df.duplicated().sum()
    print(f"\nExact duplicate rows: {dups}")

# ========================================== Missing Value Handling ==========================================

def check_missing(df, name):
    print(f"\n==== Missing Values in {name} ====")
    if df is None:
        print("Dataset not loaded.")
        return
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("No missing values")
    else:
        total = len(df)
        missing_df = pd.DataFrame({
            "missing_count": missing,
            "missing_percent": (missing / total * 100).round(2)
        })
        print(missing_df)

def missing_value_handling_books(df):
    """Handle missing values for the books dataset."""
    df = df.copy()
    
    # Fill language_code with "unknown"
    df["language_code"] = df["language_code"].fillna("unknown")

    # Fill original_title with title (if available), else "unknown"
    df["original_title"] = df.apply(
        lambda row: row["title"] if pd.isna(row["original_title"]) else row["original_title"],
        axis=1
    )
    df["original_title"] = df["original_title"].fillna("unknown")

    # Drop rows where original_publication_year is missing
    before_drop = df.shape[0]
    df = df.dropna(subset=["original_publication_year"])
    after_drop = df.shape[0]

    print(f"Books: Rows before dropping publication_year nulls: {before_drop}")
    print(f"Books: Rows after dropping: {after_drop} (removed {before_drop - after_drop})")

    return df

# ========================================== Duplicate Handling ==========================================

def report_exact_duplicates(df, name):
    """Report exact duplicate rows in a dataset"""
    if df is None:
        print(f"\n[{name}] Dataset not loaded")
        return
    exact_dups = df.duplicated().sum()
    print(f"[{name}] Exact duplicate rows: {exact_dups}")

def report_key_duplicates(df, name, key_cols):
    """Report duplicates based on key columns"""
    if df is None:
        print(f"\n[{name}] Dataset not loaded")
        return
    
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    
    # Check if all key columns exist
    missing_cols = [col for col in key_cols if col not in df.columns]
    if missing_cols:
        print(f"\n[{name}] Missing key columns: {missing_cols}")
        return
    
    key_dups = df.duplicated(subset=key_cols).sum()
    print(f"[{name}] Duplicates by {key_cols}: {key_dups}")

def duplicate_value_handling_ratings(df):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "book_id"], keep="first")
    after = len(df)
    print(f"[RATINGS] Before: {before}, After: {after}, Removed: {before - after}")
    return df

def duplicate_value_handling_book_tags(df):
    df = df.copy()
    bk_col = "book_id" if "book_id" in df.columns else "goodreads_book_id"
    before = len(df)
    df = df.drop_duplicates(subset=[bk_col, "tag_id"], keep="first")
    after = len(df)
    print(f"[BOOK_TAGS] Before: {before}, After: {after}, Removed: {before - after}")
    return df

def duplicate_value_handling_tags(df):
    df = df.copy()
    before = len(df)
    df["_norm_name"] = df["tag_name"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["_norm_name"], keep="first")
    df = df.drop(columns=["_norm_name"])
    after = len(df)
    print(f"[TAGS] Before: {before}, After: {after}, Removed: {before - after}")
    return df

# ========================================== Structural Error Handling ==========================================

def check_structural_errors():
    print("\n==== Structural Error Checks ====")

    # Data types
    print("\n[DATA TYPES]")
    print(books.dtypes)

    # Check unique values in categorical columns
    print("\n[BOOKS] language_code unique values (sample):")
    print(books["language_code"].dropna().unique()[:20])

    print("\n[TAGS] tag_name samples (possible inconsistencies):")
    print(tags["tag_name"].sample(20, random_state=42).tolist())

    # Check ratings value range
    print("\n[RATINGS] value range:")
    print(f"min={ratings['rating'].min()}, max={ratings['rating'].max()}")

    # Check year range validity
    print("\n[BOOKS] publication year range:")
    print(f"min={books['original_publication_year'].min()}, max={books['original_publication_year'].max()}")

def fix_structural_errors(datasets):
    fixed = {name: df.copy() for name, df in datasets.items()}
    
    # --- BOOKS fixes ---
    if "books" in fixed:
        books = fixed["books"]

        # Normalize language codes (only obvious ones)
        lang_map = {
            "en-US": "eng", "en-GB": "eng", "en-CA": "eng", "en": "eng",
            "fre": "fra", "ger": "deu", "spa": "spa", "por": "por", 
            "ita": "ita", "vie": "vie", "ind": "ind"  # keep rare ones as is
        }
        books["language_code"] = books["language_code"].replace(lang_map)

        # Publication year cleanup
        books["original_publication_year"] = books["original_publication_year"].apply(
            lambda y: y if (pd.notnull(y) and 1000 <= y <= 2025) else pd.NA
        )

        # ISBNs → strings, don't force formatting
        books["isbn"] = books["isbn"].astype("string")
        books["isbn13"] = books["isbn13"].astype("string")

        fixed["books"] = books

    # RATINGS fixes
    if "ratings" in fixed:
        ratings = fixed["ratings"]
        # Ratings already 1–5, nothing to fix here
        fixed["ratings"] = ratings

    return fixed

# ========================================== Outlier Filtering ==========================================

def filter_outliers(datasets):
    """Filter outliers from datasets"""
    filtered = {}
    
    for name, df in datasets.items():
        if df is None:
            filtered[name] = None
            continue
            
        df_filtered = df.copy()
        
        if name == "books":
            # Filter books with extreme publication years
            if "original_publication_year" in df_filtered.columns:
                # Keep books published between 1000 and 2025
                df_filtered = df_filtered[
                    (df_filtered["original_publication_year"] >= 1000) & 
                    (df_filtered["original_publication_year"] <= 2025)
                ]
        
        elif name == "ratings":
            # Filter ratings within valid range (1-5)
            if "rating" in df_filtered.columns:
                df_filtered = df_filtered[
                    (df_filtered["rating"] >= 1) & 
                    (df_filtered["rating"] <= 5)
                ]
        
        print(f"[{name}] Outlier filtering: {len(df)} -> {len(df_filtered)} rows")
        filtered[name] = df_filtered
    
    return filtered

# ========================================== Enhanced Text Processing ==========================================

class EnhancedTextProcessor:
    """Enhanced text processing with NLTK integration"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Enhanced text cleaning following best practices"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters, keep only letters, numbers, and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def prepare_enhanced_item_descriptions(self, books_df, book_tags_df, tags_df):
        """Create enhanced item descriptions with proper text preprocessing"""
        print(" Preparing enhanced item descriptions...")
        
        # Create book-tag mapping
        book_tag_merged = book_tags_df.merge(tags_df, on='tag_id', how='left')
        book_tag_mapping = {}
        for book_id, group in book_tag_merged.groupby('goodreads_book_id'):
            tag_data = group[['tag_name']].dropna()
            tag_names = tag_data['tag_name'].tolist()
            book_tag_mapping[book_id] = tag_names
        
        descriptions = []
        for idx, book in books_df.iterrows():
            description_parts = []
            
            # Clean and add title
            if pd.notna(book.get('title')):
                clean_title = self.clean_text(book['title'])
                if clean_title:
                    description_parts.append(clean_title)
            
            # Clean and add authors
            if pd.notna(book.get('authors')):
                clean_authors = self.clean_text(book['authors'])
                if clean_authors:
                    description_parts.append(clean_authors)
            
            # Add cleaned tags
            book_id = book['book_id']
            book_tags = book_tag_mapping.get(book_id, [])
            if book_tags:
                clean_tags = [self.clean_text(tag) for tag in book_tags if self.clean_text(tag)]
                description_parts.extend(clean_tags)
            
            # Add publication year (as text feature)
            if pd.notna(book.get('original_publication_year')):
                year = int(book['original_publication_year'])
                if 1000 <= year <= 2025:
                    decade = (year // 10) * 10
                    description_parts.append(f"published {decade}s")
            
            # Add language (as text feature)
            if pd.notna(book.get('language_code')):
                lang = str(book['language_code']).lower()
                if lang != 'unknown':
                    description_parts.append(f"language {lang}")
            
            # Combine all parts
            full_description = " ".join(description_parts)
            descriptions.append(full_description)
        
        print(f"Created {len(descriptions)} enhanced item descriptions")
        return descriptions

# ========================================== Enhanced Feature Engineering ==========================================

class EnhancedFeatureEngineer:
    """Enhanced feature engineering for recommender systems"""
    
    def __init__(self):
        self.lsa_scaler = StandardScaler()
        self.user_scaler = StandardScaler()
        self.tfidf_vectorizer = None
        self.lsa_model = None
    
    def create_enhanced_lsa_features(self, descriptions, n_components=100):
        """Create enhanced LSA features with proper preprocessing"""
        print(" Creating enhanced LSA features...")
        
        # Enhanced TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=15000,  # Increased vocabulary
            stop_words='english',
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            min_df=3,  # Increased minimum document frequency
            max_df=0.8,  # Decreased maximum document frequency
            sublinear_tf=True,
            norm='l2'  # L2 normalization
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Apply LSA (Truncated SVD)
        self.lsa_model = TruncatedSVD(
            n_components=n_components,
            random_state=42,
            algorithm='arpack'
        )
        
        lsa_features = self.lsa_model.fit_transform(tfidf_matrix)
        
        # Normalize LSA features
        lsa_features_scaled = self.lsa_scaler.fit_transform(lsa_features)
        
        # Print explained variance
        explained_variance = self.lsa_model.explained_variance_ratio_.sum()
        print(f"LSA explained variance: {explained_variance:.3f}")
        print(f"LSA features shape: {lsa_features_scaled.shape}")
        
        return lsa_features_scaled
    
    def create_enhanced_user_features(self, user_ids, ratings_df, user_features_size=15):
        """Create enhanced user features with more sophisticated engineering"""
        print(" Creating enhanced user features...")
        
        # Create user mappings
        unique_users = sorted(list(set(user_ids)))
        user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        
        user_features = []
        for user_id in user_ids:
            user_idx = user_id_map[user_id]
            user_feature_vector = []
            
            # Basic user features
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            if not user_ratings.empty:
                # Rating statistics
                avg_rating = user_ratings['rating'].mean()
                rating_std = user_ratings['rating'].std()
                rating_count = len(user_ratings)
                rating_median = user_ratings['rating'].median()
                
                # Rating distribution features
                rating_dist = user_ratings['rating'].value_counts().sort_index()
                rating_entropy = -sum((p/rating_count) * np.log2(p/rating_count) for p in rating_dist if p > 0)
                
                # Temporal features (if timestamp available)
                if 'timestamp' in user_ratings.columns:
                    user_ratings_sorted = user_ratings.sort_values('timestamp')
                    rating_frequency = rating_count / (user_ratings_sorted['timestamp'].max() - user_ratings_sorted['timestamp'].min() + 1)
                else:
                    rating_frequency = 0.0
                
                # Book diversity (unique books rated)
                unique_books = user_ratings['book_id'].nunique()
                book_diversity = unique_books / rating_count if rating_count > 0 else 0
                
                # User ID encoding (normalized)
                user_feature_vector.append(user_idx / len(unique_users))
                
                # Enhanced rating features
                user_feature_vector.extend([
                    avg_rating / 5.0,  # Normalized average rating
                    rating_std / 5.0 if pd.notna(rating_std) else 0.0,  # Normalized std
                    min(rating_count / 100.0, 1.0),  # Normalized count (capped)
                    rating_median / 5.0,  # Normalized median
                    min(rating_entropy / 3.0, 1.0),  # Normalized entropy
                    min(rating_frequency, 1.0),  # Normalized frequency
                    book_diversity,  # Book diversity
                    min(unique_books / 1000.0, 1.0)  # Normalized unique books
                ])
            else:
                # Default features for new users
                user_feature_vector.extend([0.0] * 8)
            
            # Pad or truncate to fixed size
            while len(user_feature_vector) < user_features_size:
                user_feature_vector.append(0.0)
            
            user_feature_vector = user_feature_vector[:user_features_size]
            user_features.append(user_feature_vector)
        
        # Normalize user features
        user_features = np.array(user_features)
        user_features_scaled = self.user_scaler.fit_transform(user_features)
        
        print(f"Enhanced user features shape: {user_features_scaled.shape}")
        return user_features_scaled, user_id_map

# ========================================== Main Data Preparation Pipeline ==========================================

def run_complete_data_preparation():
    """Run the complete enhanced data preparation pipeline"""
    print(" Starting Complete Enhanced Data Preparation Pipeline")
    print("=" * 70)
    
    # Step 1: Data Understanding
    print("\n STEP 1: Data Understanding")
    print("-" * 40)
    summarize_df("books", books)
    summarize_df("ratings", ratings)
    summarize_df("book_tags", book_tags)
    summarize_df("tags", tags)
    summarize_df("to_read", to_read)
    
    # Step 2: Missing Value Handling
    print("\n STEP 2: Missing Value Handling")
    print("-" * 40)
    check_missing(books, "Books")
    check_missing(ratings, "Ratings")
    check_missing(book_tags, "Book Tags")
    check_missing(tags, "Tags")
    check_missing(to_read, "To-Read")
    
    # Handle missing values
    cleaned_books = missing_value_handling_books(books)
    
    # Step 3: Duplicate Handling
    print("\n STEP 3: Duplicate Handling")
    print("-" * 40)
    report_exact_duplicates(books, "BOOKS")
    report_exact_duplicates(ratings, "RATINGS")
    report_exact_duplicates(book_tags, "BOOK_TAGS")
    report_exact_duplicates(tags, "TAGS")
    report_exact_duplicates(to_read, "TO_READ")
    
    # Handle duplicates
    cleaned_ratings = duplicate_value_handling_ratings(ratings)
    cleaned_book_tags = duplicate_value_handling_book_tags(book_tags)
    cleaned_tags = duplicate_value_handling_tags(tags)
    
    # Step 4: Structural Error Handling
    print("\n STEP 4: Structural Error Handling")
    print("-" * 40)
    check_structural_errors()
    
    # Fix structural errors
    datasets_after_structural = {
        "books": cleaned_books,
        "ratings": cleaned_ratings,
        "book_tags": cleaned_book_tags,
        "tags": cleaned_tags,
        "to_read": to_read.copy() if to_read is not None else None
    }
    
    cleaned_after_structural = fix_structural_errors(datasets_after_structural)
    
    # Step 5: Outlier Filtering
    print("\n STEP 5: Outlier Filtering")
    print("-" * 40)
    cleaned_after_outliers = filter_outliers(cleaned_after_structural)
    
    # Step 6: Enhanced Text Processing
    print("\n STEP 6: Enhanced Text Processing")
    print("-" * 40)
    text_processor = EnhancedTextProcessor()
    enhanced_descriptions = text_processor.prepare_enhanced_item_descriptions(
        cleaned_after_outliers['books'], 
        cleaned_after_outliers['book_tags'], 
        cleaned_after_outliers['tags']
    )
    
    # Step 7: Enhanced Feature Engineering
    print("\n STEP 7: Enhanced Feature Engineering")
    print("-" * 40)
    feature_engineer = EnhancedFeatureEngineer()
    
    # Create LSA features
    lsa_features = feature_engineer.create_enhanced_lsa_features(enhanced_descriptions, n_components=100)
    
    # Create user features (sample for demonstration)
    sample_users = cleaned_after_outliers['ratings']['user_id'].value_counts().head(1000).index
    user_features, user_id_map = feature_engineer.create_enhanced_user_features(
        sample_users, cleaned_after_outliers['ratings'], user_features_size=15
    )
    
    # Step 8: Final Dataset Assembly
    print("\n STEP 8: Final Dataset Assembly")
    print("-" * 40)
    
    # Add enhanced features to the cleaned datasets
    final_datasets = cleaned_after_outliers.copy()
    final_datasets['enhanced_descriptions'] = enhanced_descriptions
    final_datasets['lsa_features'] = lsa_features
    final_datasets['user_features'] = user_features
    final_datasets['user_id_map'] = user_id_map
    final_datasets['text_processor'] = text_processor
    final_datasets['feature_engineer'] = feature_engineer
    
    # Print final statistics
    print("\n FINAL DATASET STATISTICS")
    print("=" * 50)
    print(f"Books: {len(final_datasets['books']):,}")
    print(f"Ratings: {len(final_datasets['ratings']):,}")
    print(f"Users: {final_datasets['ratings']['user_id'].nunique():,}")
    print(f"Book-Tag pairs: {len(final_datasets['book_tags']):,}")
    print(f"Tags: {len(final_datasets['tags']):,}")
    print(f"LSA Features: {lsa_features.shape}")
    print(f"User Features: {user_features.shape}")
    
    print("\n Enhanced Data Preparation Complete!")
    print("All datasets are now ready for advanced recommender systems!")
    
    return final_datasets

# ========================================== Legacy Compatibility ==========================================

# For backward compatibility, keep the original cleaned_after_outliers
# This will be populated when the module is imported
cleaned_after_outliers = None

def initialize_enhanced_data():
    """Initialize enhanced data preparation"""
    global cleaned_after_outliers
    if cleaned_after_outliers is None:
        print("Loading enhanced data preparation...")
        cleaned_after_outliers = run_complete_data_preparation()
    return cleaned_after_outliers

# Lazy initialization - only run when explicitly requested
_cleaned_after_outliers = None

def get_cleaned_data():
    """Get cleaned data, initializing only when needed"""
    global _cleaned_after_outliers
    if _cleaned_after_outliers is None:
        print("Initializing data preparation (first time only)...")
        _cleaned_after_outliers = initialize_enhanced_data()
    return _cleaned_after_outliers

# For backward compatibility, but don't auto-initialize
cleaned_after_outliers = None

if __name__ == "__main__":
    cleaned_after_outliers = run_complete_data_preparation()
else:
    # Don't auto-initialize - use get_cleaned_data() when needed
    pass
