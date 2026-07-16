import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class DenseRetriever:
    """Dense retrieval over movie metadata using precomputed embeddings."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", path_embeddings=None, path_csv=None):
        """
        Initialize the embedding model and keep embeddings in memory.
        :param model_name: Hugging Face model name.
        """
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"

        if path_csv is None:
            path_csv = data_dir / "popular_movies_10k_clean.csv"

        self.df = pd.read_csv(path_csv)
        scaler = MinMaxScaler()
        self.df["popularity"] = scaler.fit_transform(self.df[["popularity"]])
        self.model = SentenceTransformer(model_name)
        
        self.embeddings = None

        # Concatenate title and overview to generate embeddings.
        self.df["text"] = self.df["title"] + " - " + self.df["overview"]

        self._generate_embeddings(path_embeddings)

    def _generate_embeddings(self, path_embeddings=None):
        """Generate embeddings and keep them in memory."""
        if path_embeddings is not None and os.path.exists(path_embeddings):
            self.embeddings = np.load(path_embeddings)
            print(f"Embeddings loaded from: {path_embeddings}")
        else:
            default_embeddings_path = Path(__file__).resolve().parent.parent / "data" / "imdb_embeddings.npy"
            path_embeddings = str(default_embeddings_path)
            if os.path.exists(path_embeddings):
                print("Removing previous embeddings...")
                os.remove(path_embeddings)
            print("Generating new embeddings...")
            self.embeddings = self.model.encode(self.df["text"].tolist(), convert_to_numpy=True)
            self.save_embeddings(path_embeddings)
            print(f"Embeddings generated and stored in {path_embeddings}.")
            

    def save_embeddings(self, path):
        """Save embeddings to a .npy file."""
        np.save(path, self.embeddings)
        print(f"Embeddings saved to: {path}")


    def search(self, query, top_k=5):
        """
        Perform a cosine-similarity search.
        :param query: Search text.
        :param top_k: Number of results to return.
        :return: DataFrame with results sorted by similarity.
        """
        print(f"🔍 Searching for: {query}")

        # Convert the query into an embedding.
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Compute cosine similarity between the query and movie embeddings.
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Retrieve the best result indices.
        best_indices = np.argsort(similarities)[::-1][:top_k]

        # Collect the matching movies.
        results = self.df.iloc[best_indices].copy()
        results["similarity"] = similarities[best_indices]

        return results.sort_values(by="similarity", ascending=False)

