

import requests
import time
import pandas as pd
import os
import argparse
from pathlib import Path
from tqdm import tqdm

"""Download popular TMDB movies for building the local movie corpus."""

BASE_URL = "https://api.themoviedb.org/3/movie/popular"
NUM_MOVIES = 10000  # Total number of movies to download.
MOVIES_PER_PAGE = 20  # TMDB returns 20 movies per page.
pages_to_download = (NUM_MOVIES // MOVIES_PER_PAGE) + 1


def get_movies(api_key, total=NUM_MOVIES):
    """Fetch the most popular movies from TMDB."""
    movies = []
    for page in tqdm(range(1, pages_to_download + 1), desc="Downloading movies"):
        url = f"{BASE_URL}?api_key={api_key}&language=en-EN&page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            movies.extend(data["results"])

        else:
            print(f"⚠ Request error: {response.status_code}")
            break
        # Pause to avoid exceeding API limits (40 requests/10s).
        time.sleep(0.1)
        # Stop once the requested limit is reached.
        if len(movies) >= total:
            break

    return movies[:total]


def save_movies_csv(movies, output_path):
    """Persist the downloaded movie list to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies_df = pd.DataFrame(movies)
    movies_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Saved {len(movies)} movies to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download popular TMDB movies and save them to a CSV file.")
    parser.add_argument("--api_key", type=str, default=os.getenv("TMDB_API_KEY"), help="TMDB API key.")
    parser.add_argument("--path_save", type=str, default="./data/popular_movies_10k.csv", help="Path where the popular movies CSV file will be saved.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("TMDB API key is required. Pass --api_key or set TMDB_API_KEY.")

    movies = get_movies(args.api_key, NUM_MOVIES)
    save_movies_csv(movies, args.path_save)


if __name__ == "__main__":
    main()