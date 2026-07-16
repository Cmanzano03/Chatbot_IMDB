from transformers import pipeline

class RAGQA:
    """Generate a short natural-language answer from retrieved movie context."""

    def __init__(self, retriever, model_name="google/flan-t5-large"):
        """
        Initialize the RAG-based question answering system.
        :param retriever: DenseRetriever instance.
        :param model_name: Hugging Face generation model.
        """
        self.retriever = retriever
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            max_length=200,
            min_length=50,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.7
        )

    def generate_answer(self, query, top_k=1):
        """
        Retrieve information and generate a natural-language answer.
        :param query: User question.
        :param top_k: Number of documents to retrieve.
        :return: Model-generated response.
        """
        retrieved_movies = self.retriever.search(query, top_k=top_k)

        context = "\n".join([
            f"- {row['title']} ({row['release_date']}): {row['overview']}"
            for _, row in retrieved_movies.iterrows()
        ])

        prompt = (
        
            f"CONTEXT:\n{context}\n\n"
            "TALKS ABOUT THE MAIN PLOT OF THE MOVIE.\n"
            "DO NOT MENTION, FILM TITLE, FILM RELEASE YEAR, DIRECTOR, ACTORS.\n"
            "FOCUS YOUR DESCRIPTION ON THE PLOT, GENRE, OR THEME. ONLY PROVIDING A SHORT RESUME.\n"
        )


        response = self.generator(prompt, max_length=100, truncation=True)
        summary = response[0]["generated_text"]
        index = summary.find(":")
        summary = summary[index + 1:]
        title = retrieved_movies.iloc[0]["title"]
        year = retrieved_movies.iloc[0]["release_date"]
        adult = retrieved_movies.iloc[0]["adult"]
       
        return summary, title, year, adult
