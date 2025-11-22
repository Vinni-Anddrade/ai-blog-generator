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

    def set_up_graph(self, usecase):
        if usecase == "topic":
            self.building_graph()
            return self.graph_model


## Structure for Langgraph studio
llm = GroqLLM().get_llm()

graph_builder = GraphBuilder(llm)
graph = graph_builder.set_up_graph(usecase="topic")
