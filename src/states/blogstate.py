from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import add_messages


class LanguageDetection(BaseModel):
    current_language: Literal["Portuguese", "Spanish", "English"] = Field(
        description="Language of the blog"
    )


class Blog(BaseModel):
    title: str = Field(description="Title of the blog")
    content: str = Field(description="The main content of the blog post")


class BlogState(TypedDict):
    topic: str
    blog: Blog
    language: LanguageDetection
