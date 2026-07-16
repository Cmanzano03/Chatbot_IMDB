"""
How to run:

pip install streamlit

streamlit run src/IMDB_chatbot_UI.py
"""

from pathlib import Path
import streamlit as st
from RAG import RAGQA
import DenseRetriever as DR

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Initialize and return the DenseRetriever and RAGQA instances.
    This function is cached globally to prevent reloading weights on subsequent runs.
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    csv_path = data_dir / "popular_movies_10k_clean.csv"
    embeddings_path = data_dir / "imdb_embeddings.npy"

    retriever = DR.DenseRetriever(path_embeddings=embeddings_path, path_csv=csv_path)
    rag = RAGQA(retriever)
    return retriever, rag


def main():
    st.set_page_config(page_title="MovieFinder", page_icon="🎬", layout="centered")

    # Injecting custom CSS for a premium, stunning dark mode and cinematic look
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        /* Dark themed gradient background */
        .stApp {
            background: radial-gradient(circle at top right, #1b0a2d, #0b0518 50%, #020105) !important;
            color: #f3f4f6 !important;
            font-family: 'Outfit', sans-serif !important;
        }

        /* Movie Finder Title with gradient text */
        h1 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #ff007f, #7f00ff, #00f0ff) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            text-align: center !important;
            font-size: 3.2rem !important;
            margin-bottom: 0.2rem !important;
            letter-spacing: -1px;
            filter: drop-shadow(0px 4px 12px rgba(127, 0, 255, 0.3));
        }

        /* Header Subtitle Banner */
        .subtitle-banner {
            background: linear-gradient(90deg, rgba(255, 0, 127, 0.15), rgba(127, 0, 255, 0.15)) !important;
            border-left: 4px solid #ff007f !important;
            border-right: 4px solid #00f0ff !important;
            color: #e5e7eb !important;
            padding: 14px 20px !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 1.1rem !important;
            text-align: center !important;
            margin-bottom: 2rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }

        /* Text Input Customization */
        .stTextInput>div>div>input {
            border-radius: 12px !important;
            padding: 14px 18px !important;
            font-size: 16px !important;
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
        }

        .stTextInput>div>div>input:focus {
            border-color: #00f0ff !important;
            box-shadow: 0 0 18px rgba(0, 240, 255, 0.25) !important;
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        /* Button Customization (Standard and Form Submit) */
        div[data-testid="stFormSubmitButton"] button, .stButton>button {
            background: linear-gradient(135deg, #7f00ff, #ff007f) !important;
            color: white !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            border-radius: 12px !important;
            padding: 14px 30px !important;
            border: none !important;
            box-shadow: 0 4px 20px rgba(127, 0, 255, 0.4) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
            cursor: pointer !important;
            letter-spacing: 0.5px;
        }

        div[data-testid="stFormSubmitButton"] button:hover, .stButton>button:hover {
            background: linear-gradient(135deg, #ff007f, #7f00ff) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px rgba(255, 0, 127, 0.6) !important;
        }

        div[data-testid="stFormSubmitButton"] button:active, .stButton>button:active {
            transform: translateY(1px) !important;
        }

        /* Results container card */
        .movie-card {
            background: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 16px !important;
            padding: 26px !important;
            margin-top: 24px !important;
            margin-bottom: 24px !important;
            box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5) !important;
            transition: all 0.3s ease !important;
        }

        .movie-card:hover {
            border-color: rgba(0, 240, 255, 0.3) !important;
            box-shadow: 0 12px 50px 0 rgba(0, 240, 255, 0.15) !important;
            transform: translateY(-3px);
        }

        /* Input labels */
        label {
            color: #d1d5db !important;
            font-weight: 500 !important;
            font-size: 1.05rem !important;
            margin-bottom: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🎬 Movie Finder")
    st.markdown(
        """
        <div class="subtitle-banner">
            Describe the plot, characters, or vibes, and our AI will retrieve the best match.
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Loading Screen (Cold Start)
    # The load_models call is wrapped in a st.spinner. Resource caching ensures
    # this spinner is only shown on the first app execution (cold start).
    with st.spinner('Loading model weights for the first time... This may take a few minutes.'):
        retriever, rag = load_models()

    # Wrap inside a form to enable "Enter" key submissions naturally
    with st.form(key="movie_finder_form", clear_on_submit=False):
        question = st.text_input("Provide a description:", "A man bitten by a spider")
        submitted = st.form_submit_button("Guess")

    # 3. Isolated Inference
    # The logic inside the form submission exclusively consumes search() and generate_answer()
    # from the cached retriever and rag instances, with absolutely no instantiation.
    if submitted:
        with st.spinner("Generating recommendation..."):
            # Execute search
            results = retriever.search(question, top_k=5)
            
            # Execute text generation
            answer, title, year, adult = rag.generate_answer(question, top_k=1)
            
            adult_str = "Yes" if adult else "No"
            year_clean = year.split('-')[0] if isinstance(year, str) and '-' in year else str(year)
            
            # Render premium movie card
            st.markdown(f"""
            <div class="movie-card">
                <h3 style="margin-top: 0; color: #00f0ff; font-weight: 700; font-size: 1.6rem; margin-bottom: 12px;">🍿 {title} ({year_clean})</h3>
                <div style="display: flex; gap: 12px; margin-bottom: 20px;">
                    <span style="background: rgba(0, 240, 255, 0.1); color: #00f0ff; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid rgba(0, 240, 255, 0.25); font-weight: 600;">
                        Top Match
                    </span>
                    <span style="background: rgba(255, 0, 127, 0.15); color: #ff007f; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid rgba(255, 0, 127, 0.25); font-weight: 600;">
                        Adult Only: {adult_str}
                    </span>
                </div>
                <p style="color: #e5e7eb; line-height: 1.7; font-size: 1.1rem; margin-bottom: 0;">
                    <strong style="color: #ff007f;">AI Plot Synthesis:</strong><br>{answer}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show other close matches in a sleek way
            if len(results) > 1:
                st.markdown("### 🔍 Other Relevant Matches")
                for idx, row in results.iloc[1:4].iterrows():
                    match_year = row['release_date'].split('-')[0] if isinstance(row['release_date'], str) and '-' in row['release_date'] else str(row['release_date'])
                    match_adult = "Yes" if row['adult'] else "No"
                    st.markdown(
                        f"- **{row['title']}** ({match_year}) — Similarity: `{row['similarity']:.3f}` | Adult: `{match_adult}`"
                    )

if __name__ == "__main__":
    main()
