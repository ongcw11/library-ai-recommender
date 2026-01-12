import numpy as np
import pandas as pd
from typing import Optional, Dict

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

from data_preparation import get_cleaned_data, EnhancedTextProcessor, EnhancedFeatureEngineer


class CollaborativeRecommender:

	def __init__(self, n_components: int = 50):
		self.n_components = n_components
		self._books = None
		self._ratings = None
	
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

		# Initialize mapping variables
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
		df = self.ratings.dropna(subset=["user_id", "book_id", "rating"]).copy()
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

	def recommend_for_user(self, user_id: int, top_n: int = 20, mode: str = "svd") -> pd.DataFrame:
		"""Return CF recommendations with columns expected by templates."""
		if mode == "knn":
			cf = self._cf_knn(user_id, top_n=top_n)
		else:
			cf = self._cf_svd(user_id, top_n=top_n)

		if cf.empty:
			return pd.DataFrame()

		meta_cols = [
			"book_id", "title", "authors", "average_rating",
			"ratings_count", "original_publication_year", "small_image_url"
		]
		meta = self.books[meta_cols]
		out = cf.merge(meta, on="book_id", how="left")
		out = out.dropna(subset=["title", "authors"]).copy()
		# map CF_score to recommendation_score (0-100)
		out["recommendation_score"] = out["CF_score"].round(1)
		# fill optional fields to keep UI clean
		if "average_rating" in out.columns:
			out["average_rating"] = out["average_rating"].fillna(0)
		if "ratings_count" in out.columns:
			out["ratings_count"] = out["ratings_count"].fillna(0)
		if "original_publication_year" in out.columns:
			out["original_publication_year"] = out["original_publication_year"].fillna(0)
		if "small_image_url" in out.columns:
			out["small_image_url"] = out["small_image_url"].fillna("")
		# sort by score with popularity tiebreakers
		out = out.sort_values(["recommendation_score", "ratings_count", "average_rating"], ascending=[False, False, False])
		return out.head(top_n)

	def _cf_knn(self, user_id: int, top_n: int = 200) -> pd.DataFrame:
		if user_id not in self.user_id_map:
			return pd.DataFrame(columns=["book_id", "CF_score"])
		user_index = self.user_id_map[user_id]
		distances, indices = self.knn.kneighbors(self.sparse_matrix[user_index], n_neighbors=min(6, len(self.user_ids)))
		neighbor_ids = [self.user_ids[i] for i in indices.flatten()][1:]
		neighbor_ratings = self.ratings[self.ratings["user_id"].isin(neighbor_ids)]
		user_rated = set(self.ratings[self.ratings["user_id"] == user_id]["book_id"].tolist())
		scored = neighbor_ratings.groupby("book_id")["rating"].mean().reset_index(name="score")
		if scored.empty:
			return pd.DataFrame(columns=["book_id", "CF_score"])
		s = scored["score"].to_numpy()
		s_min, s_max = float(s.min()), float(s.max())
		cf = 100.0 * (s - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(s)
		out = scored.assign(CF_score=cf)[["book_id", "CF_score"]]
		out = out[~out["book_id"].isin(user_rated)]
		return out.sort_values("CF_score", ascending=False).head(top_n)

	def _cf_svd(self, user_id: int, top_n: int = 200) -> pd.DataFrame:
		if user_id not in self.user_id_map:
			return pd.DataFrame(columns=["book_id", "CF_score"])
		user_index = self.user_id_map[user_id]
		user_vector = self.svd_matrix[user_index, :]
		scores = np.dot(user_vector, self.book_embeddings.T)
		rated = set(self.ratings[self.ratings["user_id"] == user_id]["book_id"].tolist())
		scores = scores.copy()
		rated_idx = [self.book_id_map[b] for b in rated if b in self.book_id_map]
		for idx in rated_idx:
			scores[idx] = -1e12
		valid = scores[scores > -1e11]
		if valid.size > 0:
			vmin, vmax = valid.min(), valid.max()
			norm = 100.0 * (scores - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(scores)
		else:
			norm = np.zeros_like(scores)
		top_indices = np.argsort(norm)[::-1][: top_n * 3]
		top_books = [self.book_ids[i] for i in top_indices]
		out = pd.DataFrame({"book_id": top_books, "CF_score": norm[top_indices]})
		out = out[out["CF_score"] > 0]
		return out.sort_values("CF_score", ascending=False).head(top_n)


