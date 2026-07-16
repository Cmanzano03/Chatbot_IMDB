"""
How to run:

pip install streamlit

streamlit run src/IMDB_chatbot_UI.py
"""

from pathlib import Path

import streamlit as st
from RAG import RAGQA
import DenseRetriever as DR

@st.cache_resource
def load_retriever(csv_path, embeddings_path):
    return DR.DenseRetriever(path_embeddings=embeddings_path, path_csv=csv_path)


@st.cache_resource
def load_rag(retriever):
    return RAGQA(retriever)

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    st.set_page_config(page_title="MovieFinder", page_icon="🎬", layout="centered")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f4f4f4;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            font-family: 'Helvetica', sans-serif;
            color: #1a73e8;
            text-align: center;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            width: 100%;
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #ddd;
        }
        .stTextInput>div>div>input:focus {
            border: 1px solid #1a73e8;
        }
        .stSelectbox>div>div>div>div>div>div>input {
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            width: 100%;
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #ddd;
        }
        .stSelectbox>div>div>div>div>div>div>input:focus {
            border: 1px solid #1a73e8;
        }
        .stWarning {
            background-color: #ffcc00;
            color: black;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton>button#recommendations_button {
            background-color: green;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
            display: block;
            margin: 0 auto;
        }
        .stButton>button#recommendations_button:hover {
            background-color: lightgreen;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🎬 Movie Finder")
    st.markdown(
        """
        <div style="background-color: green; color: white; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 18px; text-align: center;">
            Find the movie you do not remember the title.
        </div>
        """,
        unsafe_allow_html=True
    )

    question = st.text_input("Provide a description:", "A man bitten by a spider")
    
    if st.button("Guess", key="recommendations_button"):
        csv_path = data_dir / "popular_movies_10k_clean.csv"
        embeddings_path = data_dir / "imdb_embeddings.npy"

        with st.spinner("Loading retrieval and generation models..."):
            retriever = load_retriever(csv_path, embeddings_path)
            rag = load_rag(retriever)

        with st.spinner("Generating recommendation..."):
            answer, title, year, adult = rag.generate_answer(question, top_k=1)
            adult = "Yes" if adult else "No"
            st.write(f"**Movie Title:** {title} **Year:** {year.split('-')[0]} **Adult Only:** {adult}")
            st.write(answer)


if __name__ == "__main__":
    main()
