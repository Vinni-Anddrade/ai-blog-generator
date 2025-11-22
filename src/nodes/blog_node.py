from src.states.blogstate import BlogState
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class BlogNode:
    def __init__(self, llm: ChatGroq):
        self.llm = llm

    def title_creation(self, state: BlogState):
        """
        Create the title for the blog
        """
        if "topic" in state and state["topic"]:
            prompt = PromptTemplate.from_template(
                template="""
                You are an expert blog content writer. Use markdown formatting
                and generate a blog title for the {topic}.
                It must be creative and SEO friendly.
            """
            )

            chain = prompt | self.llm

            response = chain.invoke(input={"topic": state["topic"]})

        return {"blog": {"title": response.content}}

    def blog_content_creation(self, state: BlogState):
        """
        Create the content for the entire blog
        """
        if "topic" in state and state["topic"]:
            prompt = PromptTemplate.from_template(
                template="""
            You are a specialist in writing blogs and articles. Use a markdown formatting
            to create a blog based on the topic presented: {topic}
            The writing must be captivating so we can get more and more readers.
            Use all triggers necessary to make people approach into our blog.
            """
            )

            chain = prompt | self.llm

            response = chain.invoke(input={"topic": state["topic"]})

        return {"blog": {"content": response.content}}
