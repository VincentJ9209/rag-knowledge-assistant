"""Reusable retrieval-augmented generation components."""

INSTRUCTIONS = """
You answer questions from course participants using only the provided context.

Use the context to provide an accurate answer.
If the context does not contain the answer, say that you don't know.
""".strip()


USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
""".strip()


class RAGBase:
    """Coordinate retrieval, prompt construction, and LLM generation."""

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-5.6",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(self, query, num_results=5):
        """Retrieve FAQ documents relevant to the query."""
        boost_dict = {
            "question": 2.0,
            "section": 0.5,
        }

        filter_dict = {
            "course": self.course,
        }

        return self.index.search(
            query=query,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=num_results,
        )

    def build_context(self, search_results):
        """Convert retrieved FAQ documents into a prompt-ready context string."""
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        """Build a user prompt from the query and retrieved documents."""
        context = self.build_context(search_results)

        return self.prompt_template.format(
            question=query,
            context=context,
        ).strip()

    def llm(self, user_prompt):
        """Generate an answer from the configured LLM."""
        message_history = [
            {
                "role": "developer",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=message_history,
        )

        return response.output_text

    def rag(self, query):
        """Run the complete retrieval-augmented generation pipeline."""
        search_results = self.search(query)
        user_prompt = self.build_prompt(query, search_results)
        answer = self.llm(user_prompt)

        return answer