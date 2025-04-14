import logging
import os
import yaml
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from datasets import load_dataset, load_from_disk
import sys
import json
import torch
import re
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ds_code = Dataset.load_from_disk("/Users/ssuresh/aiml/code_porting_models/dataset/babeltower")
ds_code_train = ds_code.select(range(100))
logger.info(f"Dataset: {ds_code}")

@dataclass
class CodeChunk:
    action_type: str
    ast_node: str
    code_snippet: str

# Define model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

def print_tree(node, code, indent=0):
    """Recursively print the tree structure."""
    print("  " * indent + f"{node.type}: {code[node.start_byte:node.end_byte]}")
    for child in node.children:
        print_tree(child, code, indent + 1)

def parse_cpp_code(code):
    """Parse C++ code using Tree-sitter."""
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    print_tree(root_node, code)
    return root_node

code = ds_code_train[4]['code']
root = parse_cpp_code(code)

def extract_action_nodes(node, code_bytes, actions):
    if node.type == 'for_statement':
        actions.append({
            "action": "parallelize_for_loop",
            "node": node,
            "snippet": code_bytes[node.start_byte:node.end_byte]
        })
    elif node.type == 'function_definition':
        actions.append({
            "action": "parallelize_function",
            "node": node,
            "snippet": code_bytes[node.start_byte:node.end_byte]
        })

    for child in node.children:
        extract_action_nodes(child, code_bytes, actions)

actions = []
extract_action_nodes(root, code, actions)

for act in actions:
    print(f"[{act['action']}]: {act['snippet']}")
