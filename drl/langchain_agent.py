from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from typing import List, Dict, Any
import sqlite3
import subprocess
import uuid

# ---- Mock functions (replace with actual logic) ----

def read_dataset():
    # Replace with DB read or CSV parser
    return {"input": "Sort a list of integers"}

def create_prompt(dataset_row):
    return f"Write a Python function to {dataset_row['input']}."

class CodeOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Extracts code blocks (basic)
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        return text.strip()

def compile_and_run(code: str) -> Dict[str, Any]:
    filename = f"/tmp/{uuid.uuid4().hex}.py"
    with open(filename, 'w') as f:
        f.write(code)

    try:
        result = subprocess.run(["python", filename], capture_output=True, text=True, timeout=5)
        return {"success": result.returncode == 0, "output": result.stdout}
    except Exception as e:
        return {"success": False, "output": str(e)}

# ---- LangGraph Nodes ----

def dataset_node(state):
    dataset = read_dataset()
    return {"dataset": dataset}

def prompt_node(state):
    prompt = create_prompt(state["dataset"])
    return {"prompt": prompt}

def model_inference_node(state):
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    outputs = [llm.predict(state["prompt"]) for _ in range(5)]  # Multiple generations
    return {"raw_outputs": outputs}

def parse_code_node(state):
    parser = CodeOutputParser()
    code_blocks = [parser.parse(out) for out in state["raw_outputs"]]
    return {"code_blocks": code_blocks}

def execute_code_node(state):
    results = [compile_and_run(code) for code in state["code_blocks"]]
    for i, res in enumerate(results):
        if res["success"]:
            return {"success": True, "result_index": i, "result": res}
    return {"success": False}

def rank_outputs_node(state):
    return {"final_output": state["raw_outputs"][state["result_index"]]}

# ---- Build LangGraph ----

from langgraph.graph import StateGraph

builder = StateGraph()

# Nodes
builder.add_node("read_dataset", dataset_node)
builder.add_node("create_prompt", prompt_node)
builder.add_node("model_inference", model_inference_node)
builder.add_node("parse_code", parse_code_node)
builder.add_node("execute_code", execute_code_node)
builder.add_node("rank_outputs", rank_outputs_node)

# Edges
builder.set_entry_point("read_dataset")
builder.add_edge("read_dataset", "create_prompt")
builder.add_edge("create_prompt", "model_inference")
builder.add_edge("model_inference", "parse_code")
builder.add_edge("parse_code", "execute_code")
builder.add_conditional_edges("execute_code", lambda state: "rank_outputs" if state["success"] else "model_inference")
builder.add_edge("rank_outputs", END)

# Compile graph
graph = builder.compile()

# ---- Run it ----
if __name__ == "__main__":
    final_state = graph.invoke({})
    print("🏁 Final Output:\n", final_state["final_output"])