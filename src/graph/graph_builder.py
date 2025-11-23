from langgraph.graph import StateGraph, START, END
from src.llm.groq_llm import GroqLLM
from src.states.blogstate import BlogState
from src.nodes.blog_node import BlogNode


class GraphBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(BlogState)
        self.blog_obj = BlogNode(self.llm)

    def building_graph(self):
        ## Nodes
        self.graph.add_node("title_creation", self.blog_obj.title_creation)
        self.graph.add_node("content_generator", self.blog_obj.blog_content_creation)

        ## Edges
        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generator")
        self.graph.add_edge("content_generator", END)

        self.graph_model = self.graph.compile()

    def building_graph_with_language(self):
        path_map = {
            "Portuguese": "translate_blog",
            "Spanish": "translate_blog",
            "English": END,
        }

        ## Nodes
        self.graph.add_node("title_creation", self.blog_obj.title_creation)
        self.graph.add_node("content_generator", self.blog_obj.blog_content_creation)
        self.graph.add_node("language_recognition", self.blog_obj.language_recognition)
        self.graph.add_node("translate_blog", self.blog_obj.translate_blog)
        ## Edges
        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generator")
        self.graph.add_edge("content_generator", "language_recognition")
        self.graph.add_conditional_edges(
            "language_recognition",
            self.blog_obj.route_to_translation,
            path_map=path_map,
        )

        self.graph_model = self.graph.compile()
        png_bytes = self.graph_model.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)

    def set_up_graph(self, usecase):
        if usecase == "topic":
            self.building_graph_with_language()
            return self.graph_model


## Structure for Langgraph studio
llm = GroqLLM().get_llm()

graph_builder = GraphBuilder(llm)
graph = graph_builder.set_up_graph(usecase="topic")
