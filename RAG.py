
from transformers import pipeline
import pandas as pd

class RAGQA:
    def __init__(self, retriever, model_name="google/flan-t5-base"):
        """
        Inicializa el sistema de Question Answering con RAG.
        :param retriever: Instancia de DenseRetriever.
        :param model_name: Modelo de generación de Hugging Face.
        """
        self.retriever = retriever
        self.generator = pipeline("text2text-generation", model=model_name)

    def generate_answer(self, query, top_k=3):
        """
        Recupera información y genera una respuesta en lenguaje natural.
        :param query: Pregunta del usuario.
        :param top_k: Número de documentos a recuperar.
        :return: Respuesta generada por el modelo.
        """
        #  Recuperar información con Dense Retriever
        retrieved_movies = self.retriever.search(query, top_k=top_k)

        #  Construir el contexto
        context = "\n".join([
            f"- {row['title']} ({row['release_date']}): {row['overview']}"
            for _, row in retrieved_movies.iterrows()
        ])

        #  Construir el prompt para el modelo generativo
        prompt = (
            f"Pregunta: {query}\n"
            f"Contexto:\n{context}\n"
            "Basándote en la información dada, responde de forma clara y concisa."
        )

        # Generar la respuesta
        response = self.generator(prompt, max_length=100, truncation=True)

        return response[0]["generated_text"]
