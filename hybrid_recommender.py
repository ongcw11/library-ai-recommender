import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from data_preparation import get_cleaned_data, EnhancedTextProcessor, EnhancedFeatureEngineer
from content_lsa_recommender import FinalOptimizedRecommender

# Lazy loading - data will be loaded when needed
def get_data():
    """Get data using lazy loading"""
    data = get_cleaned_data()
    return data['books'], data['ratings'], data['book_tags'], data['tags']


class CollaborativeFilteringModels:
    def __init__(self, ratings_df: pd.DataFrame, n_components: int = 50):
        self.ratings_df = ratings_df.copy()
        self.n_components = n_components

        self.user_id_map: Dict[int, int] = {}
        self.book_id_map: Dict[int, int] = {}
        self.user_ids: list[int] = []
        self.book_ids: list[int] = []

        self.sparse_matrix = None
        self.knn: Optional[NearestNeighbors] = None
        self.svd: Optional[TruncatedSVD] = None
        self.svd_matrix: Optional[np.ndarray] = None
        self.book_embeddings: Optional[np.ndarray] = None

        self._prepare()

    def _prepare(self) -> None:
        df = self.ratings_df.dropna(subset=["user_id", "book_id", "rating"]).copy()
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
        df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce").astype("Int64")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["user_id", "book_id", "rating"]).astype({"user_id": int, "book_id": int})

        self.user_ids = sorted(df["user_id"].unique().tolist())
        self.book_ids = sorted(df["book_id"].unique().tolist())
        self.user_id_map = {u: i for i, u in enumerate(self.user_ids)}
        self.book_id_map = {b: i for i, b in enumerate(self.book_ids)}

        from scipy.sparse import csr_matrix
        row = df["user_id"].map(self.user_id_map)
        col = df["book_id"].map(self.book_id_map)
        data = df["rating"].astype(np.float32)
        self.sparse_matrix = csr_matrix((data, (row, col)), shape=(len(self.user_ids), len(self.book_ids)))

        self.knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.knn.fit(self.sparse_matrix)

        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd_matrix = self.svd.fit_transform(self.sparse_matrix)
        self.book_embeddings = self.svd.components_.T

    def get_cf_scores_knn(self, user_id: int, top_n: int = 200) -> pd.DataFrame:
        if user_id not in self.user_id_map:
            return pd.DataFrame(columns=["book_id", "CF_score"])  # cold-start

        user_index = self.user_id_map[user_id]
        distances, indices = self.knn.kneighbors(self.sparse_matrix[user_index], n_neighbors=min(6, len(self.user_ids)))
        neighbor_ids = [self.user_ids[i] for i in indices.flatten()][1:]

        neighbor_ratings = self.ratings_df[self.ratings_df["user_id"].isin(neighbor_ids)]
        user_rated_books = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["book_id"].tolist())

        scored = neighbor_ratings.groupby("book_id")["rating"].mean().reset_index(name="score")
        if scored.empty:
            return pd.DataFrame(columns=["book_id", "CF_score"])  # no neighbors

        s = scored["score"].to_numpy()
        s_min, s_max = float(s.min()), float(s.max())
        if s_max > s_min:
            cf = 100.0 * (s - s_min) / (s_max - s_min)
        else:
            cf = np.zeros_like(s)

        out = scored.assign(CF_score=cf)[["book_id", "CF_score"]]
        out = out[~out["book_id"].isin(user_rated_books)]
        return out.sort_values("CF_score", ascending=False).head(top_n)

    def get_cf_scores_svd(self, user_id: int, top_n: int = 200) -> pd.DataFrame:
        if user_id not in self.user_id_map:
            return pd.DataFrame(columns=["book_id", "CF_score"])  # cold-start

        user_index = self.user_id_map[user_id]
        user_vector = self.svd_matrix[user_index, :]
        scores = np.dot(user_vector, self.book_embeddings.T)

        rated = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["book_id"].tolist())
        scores = scores.copy()
        rated_idx = [self.book_id_map[b] for b in rated if b in self.book_id_map]
        for idx in rated_idx:
            scores[idx] = -1e12

        valid = scores[scores > -1e11]
        if valid.size > 0:
            vmin, vmax = valid.min(), valid.max()
            if vmax > vmin:
                norm = 100.0 * (scores - vmin) / (vmax - vmin)
            else:
                norm = np.zeros_like(scores)
        else:
            norm = np.zeros_like(scores)

        top_indices = np.argsort(norm)[::-1][: top_n * 3]
        top_books = [self.book_ids[i] for i in top_indices]
        out = pd.DataFrame({"book_id": top_books, "CF_score": norm[top_indices]})
        out = out[out["CF_score"] > 0]
        return out.sort_values("CF_score", ascending=False).head(top_n)


class HybridRecommender:
    def __init__(self, weight_cf: float = 0.6, weight_cbf: float = 0.4, svd_components: int = 50):
        self.weight_cf = weight_cf
        self.weight_cbf = weight_cbf
        self.svd_components = svd_components
        self._cf = None
        self._cbf = None
    
    @property
    def cf(self):
        """Lazy load collaborative filtering model"""
        if self._cf is None:
            books, ratings, book_tags, tags = get_data()
            self._cf = CollaborativeFilteringModels(ratings, n_components=self.svd_components)
        return self._cf
    
    @property
    def cbf(self):
        """Lazy load content-based filtering model"""
        if self._cbf is None:
            books, ratings, book_tags, tags = get_data()
            self._cbf = FinalOptimizedRecommender(
                books, ratings, book_tags, tags, n_components=100, precision_k=10
            )
        return self._cbf

    def get_cf_scores(self, user_id: int, top_n: int = 200, mode: str = "svd") -> pd.DataFrame:
        if mode == "knn":
            return self.cf.get_cf_scores_knn(user_id, top_n=top_n)
        return self.cf.get_cf_scores_svd(user_id, top_n=top_n)

    def get_cbf_scores(self, user_id: int, top_n: int = 200) -> pd.DataFrame:
        recs = self.cbf.recommend_for_user(user_id, n_recommendations=top_n, use_percentage=True)
        if recs is None or recs.empty:
            return pd.DataFrame(columns=["book_id", "CBF_score"])
        out = recs[["book_id", "recommendation_score"]].copy()
        out = out.rename(columns={"recommendation_score": "CBF_score"})
        return out

    @staticmethod
    def combine_scores(cf_scores: pd.DataFrame, cbf_scores: pd.DataFrame, weight_cf: float = 0.6, weight_cbf: float = 0.4) -> pd.DataFrame:
        cf_use = cf_scores[["book_id", "CF_score"]] if not cf_scores.empty else pd.DataFrame(columns=["book_id", "CF_score"])
        cbf_use = cbf_scores[["book_id", "CBF_score"]] if not cbf_scores.empty else pd.DataFrame(columns=["book_id", "CBF_score"])
        merged = pd.merge(cbf_use, cf_use, on="book_id", how="outer")
        merged["CF_score"] = merged["CF_score"].fillna(0.0)
        merged["CBF_score"] = merged["CBF_score"].fillna(0.0)
        merged["final_score"] = weight_cf * merged["CF_score"] + weight_cbf * merged["CBF_score"]
        return merged.sort_values("final_score", ascending=False)

    def get_hybrid_recommendations(self, user_id: int, top_n: int = 10, mode: str = "svd") -> pd.DataFrame:
        cf_scores = self.get_cf_scores(user_id, top_n=top_n * 200, mode=mode)
        cbf_scores = self.get_cbf_scores(user_id, top_n=top_n * 200)

        if cf_scores.empty and cbf_scores.empty:
            popular = books.dropna(subset=["ratings_count", "average_rating"]).copy()
            if "ratings_count" in popular.columns:
                popular = popular.sort_values(["ratings_count", "average_rating"], ascending=[False, False])
            else:
                popular = popular.sort_values(["average_rating"], ascending=[False])
            out = popular[["book_id", "title", "authors", "average_rating", "ratings_count", "original_publication_year", "small_image_url"]].head(top_n).copy()
            out["CF_score"] = 0.0
            out["CBF_score"] = 0.0
            out["final_score"] = out["average_rating"].fillna(0) * 20.0
            return out

        combined = self.combine_scores(cf_scores, cbf_scores, self.weight_cf, self.weight_cbf)

        # Include metadata required by templates
        meta_cols = [
            "book_id", "title", "authors", "average_rating",
            "ratings_count", "original_publication_year", "small_image_url"
        ]
        meta = books[meta_cols]
        combined = combined.merge(meta, on="book_id", how="left")
        cols = [
            "book_id", "title", "authors", "average_rating",
            "ratings_count", "original_publication_year", "small_image_url",
            "CF_score", "CBF_score", "final_score"
        ]
        combined = combined[cols]
        return combined.sort_values("final_score", ascending=False).head(top_n)

    @staticmethod
    def _precision_recall_f1_at_k(recommended: pd.Series, relevant: pd.Series, k: int) -> Tuple[float, float, float]:
        if recommended.empty:
            return 0.0, 0.0, 0.0
        recommended_k = set(recommended.head(k).tolist())
        relevant_set = set(relevant.tolist())
        tp = len(recommended_k & relevant_set)
        precision = tp / max(1, len(recommended_k))
        recall = tp / max(1, len(relevant_set))
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def evaluate(self, k: int = 10, test_size: float = 0.2, random_state: int = 42, mode: str = "svd") -> pd.DataFrame:
        df = ratings.dropna(subset=["user_id", "book_id", "rating"]).copy()
        df["user_id"] = df["user_id"].astype(int)
        df["book_id"] = df["book_id"].astype(int)

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        cf_train = CollaborativeFilteringModels(train_df, n_components=self.cf.n_components)

        def get_cf_only(user_id: int) -> pd.DataFrame:
            return cf_train.get_cf_scores_svd(user_id, top_n=1000) if mode == "svd" else cf_train.get_cf_scores_knn(user_id, top_n=1000)

        results = []
        test_users = test_df["user_id"].value_counts()
        test_users = test_users[test_users >= 1].index.tolist()

        for uid in test_users:
            user_test = test_df[(test_df["user_id"] == uid) & (test_df["rating"] >= 4)]["book_id"]

            cf_only = get_cf_only(uid)
            p_cf, r_cf, f_cf = self._precision_recall_f1_at_k(cf_only["book_id"] if not cf_only.empty else pd.Series([], dtype=int), user_test, k)

            cbf_only = self.get_cbf_scores(uid, top_n=1000)
            p_cbf, r_cbf, f_cbf = self._precision_recall_f1_at_k(cbf_only["book_id"] if not cbf_only.empty else pd.Series([], dtype=int), user_test, k)

            hybrid = self.combine_scores(cf_only, cbf_only, self.weight_cf, self.weight_cbf)
            p_h, r_h, f_h = self._precision_recall_f1_at_k(hybrid["book_id"], user_test, k)

            results.append({
                "user_id": uid,
                "precision_cf": p_cf, "recall_cf": r_cf, "f1_cf": f_cf,
                "precision_cbf": p_cbf, "recall_cbf": r_cbf, "f1_cbf": f_cbf,
                "precision_hybrid": p_h, "recall_hybrid": r_h, "f1_hybrid": f_h,
            })

        res_df = pd.DataFrame(results)
        return res_df.mean(numeric_only=True).to_frame(name="score")


def get_hybrid_recommendations(user_id: int, top_n: int = 10, weight_cf: float = 0.6, weight_cbf: float = 0.4, mode: str = "svd") -> pd.DataFrame:
    recommender = HybridRecommender(weight_cf=weight_cf, weight_cbf=weight_cbf)
    return recommender.get_hybrid_recommendations(user_id, top_n=top_n, mode=mode)


