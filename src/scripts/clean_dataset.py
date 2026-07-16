import argparse
from pathlib import Path
import pandas as pd


def clean_movie_dataset(input_path, output_path=None, chunksize=1000):
    """Clean a raw TMDB export and write the compact dataset used by the app."""
    # Read the CSV file in chunks.
    chunks = pd.read_csv(input_path, encoding="utf-8", sep=",", chunksize=chunksize)

    # Concatenate chunks into a single DataFrame.
    movies_df = pd.concat(chunks, ignore_index=True)

    columns = ["title", "release_date", "popularity", "original_language", "overview", "genre_ids", "adult", "poster_path"]
    movies_df = movies_df[columns]

    # Remove duplicate rows based on all columns.
    movies_df.drop_duplicates(inplace=True)

    # Remove rows with missing values (NaN) in any column.
    movies_df.dropna(inplace=True)

    # Reset the index after removing rows.
    movies_df.reset_index(drop=True, inplace=True)

    # Save the cleaned DataFrame to a CSV file.
    if output_path is None:
        output_path = Path(input_path).with_name(f"{Path(input_path).stem}_clean.csv")
    movies_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ CSV file saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and process a raw TMDB CSV export.")
    parser.add_argument("input_path", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = input_path.with_name(f"{input_path.stem}_clean.csv")
    clean_movie_dataset(input_path, output_path)


