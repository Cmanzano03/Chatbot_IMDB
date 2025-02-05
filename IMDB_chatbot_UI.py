"""
How to run:

pip install streamlit

(remember to add the Path)

streamlit run IMDB_chatbot_UI.py
"""

import streamlit as st
import pandas as pd
from RAG import RAGQA
import DenseRetriever as DR

# Función para simular la respuesta del chatbot (por ejemplo, basado en IMDB)
def chatbot_response(query, genre):
    return f"**Mostrando resultados para la solicitud:** '{query}' **en el género:** '{genre}'."

# Interfaz de usuario de Streamlit
def main():
    # Estilo de fondo, texto y otros ajustes de apariencia (Modo claro)
    st.set_page_config(page_title="Chatbot de Películas", page_icon="🎬", layout="centered")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f4f4f4;  /* Fondo blanco */
            color: #333333;  /* Color de texto oscuro */
            font-family: 'Arial', sans-serif;
        }
        h1 {
            font-family: 'Helvetica', sans-serif;
            color: #1a73e8;  /* Título de color azul */
            text-align: center;
        }
        .stButton>button {
            background-color: #1a73e8;  /* Botón azul */
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0c55b2;  /* Botón al pasar el mouse */
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            width: 100%;
            background-color: #ffffff;  /* Fondo blanco de los cuadros de texto */
            color: #333333;  /* Color de texto oscuro */
            border: 1px solid #ddd;  /* Borde gris claro */
        }
        .stTextInput>div>div>input:focus {
            border: 1px solid #1a73e8;  /* Borde azul cuando está seleccionado */
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
            background-color: #ffcc00;  /* Fondo amarillo para advertencia */
            color: black;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    

    st.title("🎬 Movie Finder")
    st.write("This is a simple movie recommendation system based on the IMDb dataset.")
    st.write("Ask any question related to movies and I will do my best to provide you with relevant recommendations.")
    question = st.text_input("Ask me a question:")
    if st.button("Get Recommendations"):
        # Load the dataset
        pathCSV="./peliculasPopulares10k_CLEAN.csv"
        pathEmbeddings="./imdb_embeddings.npy"

        # Initialize the Dense Retriever
        retriever = DR.DenseRetriever(pathEmbeddings=pathEmbeddings, pathCSV=pathCSV)

        # Initialize the RAG-based QA system
        rag = RAGQA(retriever)

        # Generate the answer
        answer = rag.generate_answer(question, top_k=1)
        st.write(answer)
    
if __name__ == "__main__":
    main()
   
