# === Data Loading Module ===
# This module handles loading and preprocessing of the Goodreads dataset

import os
import pandas as pd

# 1) Paths
DATA_DIR = "./data"  
PATHS = {
    "books": os.path.join(DATA_DIR, "books.csv"),
    "ratings": os.path.join(DATA_DIR, "ratings.csv"),
    "book_tags": os.path.join(DATA_DIR, "book_tags.csv"),
    "tags": os.path.join(DATA_DIR, "tags.csv"),
    "to_read": os.path.join(DATA_DIR, "to_read.csv"),
}

# Safe loader
def load_csv_safe(path, **kwargs):
    """Safely load CSV files with error handling"""
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        # Try with Python engine for problematic files
        try:
            print(f"[INFO] Retrying {path} with Python engine...")
            return pd.read_csv(path, engine='python', **kwargs)
        except Exception as e2:
            print(f"[ERROR] Failed to read {path} with Python engine: {e2}")
            return None

def load_and_preprocess_data():
    
    # Load datasets 
    books = load_csv_safe(PATHS["books"])
    ratings = load_csv_safe(PATHS["ratings"])
    book_tags = load_csv_safe(PATHS["book_tags"])
    tags = load_csv_safe(PATHS["tags"])
    to_read = load_csv_safe(PATHS["to_read"])

    # Normalize key columns 
    if books is not None:
        if "book_id" in books.columns:
            books = books.rename(columns={"book_id": "book_id"})
        elif "id" in books.columns:
            books = books.rename(columns={"id": "book_id"})
        if "book_id" in books.columns:
            books["book_id"] = pd.to_numeric(books["book_id"], errors="coerce").astype("Int64")

    if ratings is not None:
        for c in ["book_id", "user_id", "rating"]:
            if c in ratings.columns:
                if c == "rating":
                    ratings[c] = pd.to_numeric(ratings[c], errors="coerce")
                else:
                    ratings[c] = pd.to_numeric(ratings[c], errors="coerce").astype("Int64")

    if book_tags is not None:
        if "goodreads_book_id" in book_tags.columns:
            book_tags["goodreads_book_id"] = pd.to_numeric(book_tags["goodreads_book_id"], errors="coerce").astype("Int64")
        if "book_id" in book_tags.columns:
            book_tags["book_id"] = pd.to_numeric(book_tags["book_id"], errors="coerce").astype("Int64")
        if "tag_id" in book_tags.columns:
            book_tags["tag_id"] = pd.to_numeric(book_tags["tag_id"], errors="coerce").astype("Int64")

    if tags is not None and "tag_id" in tags.columns:
        tags["tag_id"] = pd.to_numeric(tags["tag_id"], errors="coerce").astype("Int64")

    if to_read is not None:
        for c in ["user_id", "book_id"]:
            if c in to_read.columns:
                to_read[c] = pd.to_numeric(to_read[c], errors="coerce").astype("Int64")

    # Handle book_tags linkage if needed
    if book_tags is not None:
        if "book_id" not in book_tags.columns and "goodreads_book_id" in book_tags.columns and books is not None:
            if "goodreads_book_id" in books.columns:
                map_df = books[["book_id", "goodreads_book_id"]].dropna().drop_duplicates()
                book_tags = book_tags.merge(map_df, on="goodreads_book_id", how="left")
            else:
                print("\n[INFO] books.csv has no goodreads_book_id; cannot align book_tags yet.")

    return books, ratings, book_tags, tags, to_read

# Load data when module is imported
books, ratings, book_tags, tags, to_read = load_and_preprocess_data()
