# === Stage 1: Data Collection & Understanding (Goodbooks-10k) ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import data from the data loader module
from data_loader import books, ratings, book_tags, tags, to_read

# Quick summaries
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

summarize_df("books", books)
summarize_df("ratings", ratings)
summarize_df("book_tags", book_tags)
summarize_df("tags", tags)
summarize_df("to_read", to_read)

# Integrity checks & basic KPI
if books is not None:
    n_books = books["book_id"].nunique() if "book_id" in books.columns else len(books)
    print(f"\n[Books] Unique book_id: {n_books}")

if ratings is not None:
    n_users = ratings["user_id"].nunique() if "user_id" in ratings.columns else None
    n_books_r = ratings["book_id"].nunique() if "book_id" in ratings.columns else None
    n_ratings = len(ratings)

    print(f"\n[Ratings] rows: {n_ratings:,}")
    print(f"[Ratings] unique users: {n_users:,}" if n_users is not None else "")
    print(f"[Ratings] unique books: {n_books_r:,}" if n_books_r is not None else "")

    if "rating" in ratings.columns:
        print("\n[Ratings] basic stats:")
        print(ratings["rating"].describe())

    if n_users and n_books_r:
        sparsity = 1 - (n_ratings / (n_users * n_books_r))
        print(f"\n[Ratings] matrix sparsity ≈ {sparsity:.4f} (1.0 means very sparse)")

# Linkage checks
if (
    ratings is not None
    and books is not None
    and "book_id" in ratings.columns
    and "book_id" in books.columns
):
    matched = ratings["book_id"].isin(books["book_id"]).mean()
    print(f"\n[Link] % of ratings with a book present in books.csv: {matched*100:.2f}%")

if book_tags is not None:
    if "tag_id" in book_tags.columns:
        top_tag_counts = book_tags.groupby("tag_id")["count"].sum().sort_values(ascending=False).head(10)
        if tags is not None and "tag_name" in tags.columns:
            top_tag_counts = (
                top_tag_counts.reset_index()
                .merge(tags, on="tag_id", how="left")
                .rename(columns={"count": "total_count"})
                .sort_values("total_count", ascending=False)
            )
            print("\n[Tags] Top 10 tags by total applications:")
            print(top_tag_counts[["tag_name", "total_count"]].head(10))
        else:
            print("\n[Tags] Top 10 tag_id by total applications:")
            print(top_tag_counts)

print("\nStage 1 complete: data loaded, inspected, and profiled.")