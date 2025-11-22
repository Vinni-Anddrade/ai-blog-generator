import uvicorn
from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv

from src.graph.graph_builder import GraphBuilder
from src.llm.groq_llm import GroqLLM

load_dotenv()


app = FastAPI()


## ----------------------- ##
##            APIs         ##
## ----------------------- ##


@app.post("/blogs")
async def create_blogs(request: Request):
    data = await request.json()
    topic = data.get("topic", "")

    llm_obj = GroqLLM()
    llm_model = llm_obj.get_llm()

    # Starting the Graph
    graph_builder = GraphBuilder(llm_model)
    if topic:
        graph_builder.set_up_graph(usecase="topic")
        graph = graph_builder.graph_model

        response = graph.invoke(input={"topic": topic})

    return {"data": response}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
