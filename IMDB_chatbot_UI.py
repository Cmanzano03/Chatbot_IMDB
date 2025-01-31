"""
How to run:

pip install streamlit

(remember to add the Path)

streamlit run IMDB_chatbot_UI.py


"""


import streamlit as st

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
    
    # Título de la aplicación
    st.title('🎬 Chatbot de Películas')

    # Cuadro de texto para escribir la solicitud
    query = st.text_input("¿Qué película te gustaría buscar?", placeholder="Escribe el título de una película...")

    # Dropdown para elegir el género (con un pequeño ajuste para mejorar su apariencia)
    genres = ['Acción', 'Comedia', 'Drama', 'Terror', 'Ciencia Ficción', 'Romántica']
    genre = st.selectbox('Selecciona el género', genres)

    # Botón para enviar la solicitud
    if st.button('Buscar'):
        if query:
            response = chatbot_response(query, genre)
            # Mostrar la respuesta en formato Markdown
            st.markdown(response)
        else:
            st.warning('⚠️ Por favor ingresa una solicitud para buscar.')

# Ejecutar la interfaz
if __name__ == '__main__':
    main()

