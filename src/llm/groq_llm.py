from langchain_groq import ChatGroq
from dotenv import load_dotenv


class GroqLLM:
    def __init__(self):
        load_dotenv()

    def get_llm(self):
        model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
        return model
