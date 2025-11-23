from src.states.blogstate import BlogState
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from src.states.blogstate import LanguageDetection


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
                You are an expert blog title creator with deep knowledge of SEO, storytelling, and digital content strategy.
                Your task is to generate a single, highly creative and SEO-optimized blog title based on the following topic: \n{topic}.\n\n

                Instructions:
                Output only the blog title in Markdown format.
                The title must be engaging, curiosity-driven, memorable, and keyword-optimized.
                Avoid clichés and ensure the title stands out while remaining relevant to the topic.
                Do not generate subtitles, descriptions, or content — only the title.
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
            You are a specialist blog writer with expertise in producing engaging, high-impact blog articles.
            Your task is to write the full blog content in Markdown, based strictly on the following topic: \n{topic}.\n\n

            Instructions:
            Do not create a title. Another agent will handle that.
            Focus on captivating, emotionally engaging, and highly readable writing.
            Use storytelling, curiosity triggers, open loops, relatable examples, sensory language, and reader-connection techniques to maximize engagement.
            Ensure the blog is well structured, with clear sections, smooth transitions, and a natural flow.
            Maintain a tone that encourages readers to stay, explore, and return to the blog.
            Output only the Markdown-formatted blog content, nothing else.
            """
            )

            chain = prompt | self.llm

            response = chain.invoke(input={"topic": state["topic"]})

        return {"blog": {"title": state["blog"]["title"], "content": response.content}}

    def language_recognition(self, state):
        """
        Iditify the language of the question asked
        """
        if "topic" in state and state["topic"]:
            prompt = PromptTemplate.from_template(
                template="""
            You must detect the language of the text below.

            Return ONLY ONE WORD and it must be EXACTLY one of these options:
            - Portuguese
            - Spanish
            - English

            Topic: {topic}
            """
            )

            strucuted_llm = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0,
            )

            chain = prompt | strucuted_llm.with_structured_output(LanguageDetection)

            result: LanguageDetection = chain.invoke(input={"topic": state["topic"]})
            return {"language": result.current_language}

    def route_to_translation(self, state):
        if "language" in state and state["language"]:
            language = state["language"]
            if language != "English":
                return language
            return "English"

    def translate_title(self, state):
        """
        Make translation of a title based on the language recognized on the question or topic
        """
        if "topic" in state and state["topic"]:
            if state["language"]:
                prompt = PromptTemplate.from_template(
                    template="""
                You are the official title translator for an important blog publishing system.
                You will receive a title and a target language.
                Your task is to translate the title into the specified language.

                Instructions:
                Provide only one translated version of the title.
                Output only the translated title, with no explanations, no formatting, and no additional text.
                If the title is already in the target language or there is nothing relevant to translate, return it unchanged.

                Inputs:
                title: {title}
                language: {current_language}
                """
                )

                chain = prompt | self.llm

                title = state["blog"]["title"]
                current_language = state["language"]

                response = chain.invoke(
                    input={"title": title, "current_language": current_language}
                )

                return response.content

    def translate_content(self, state):
        """
        Make translation of a content based on the language recognized on the question or topic
        """
        if "topic" in state and state["topic"]:
            if state["language"]:
                prompt = PromptTemplate.from_template(
                    template="""
                You are the official content translator for a high-importance blog publishing system.
                You will receive a content body and a target language.
                Your task is to translate the content into the specified language.

                Instructions:
                Provide only one translated version of the content.
                Output only the translated content, with no explanations or additional text.
                If the content is already in the target language or there is nothing to translate, return it exactly as provided.

                Inputs:
                content: {content}
                language: {current_language}
                """
                )

                chain = prompt | self.llm

                content = state["blog"]["content"]
                current_language = state["language"]

                response = chain.invoke(
                    input={"content": content, "current_language": current_language}
                )

                return response.content

    def translate_blog(self, state):
        """
        Make the intire translation of the blog
        """

        title = self.translate_title(state)
        content = self.translate_content(state)

        return {"blog": {"title": title, "content": content}}
