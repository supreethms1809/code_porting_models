import tree_sitter_cpp as tscpp
import tree_sitter_cuda as tscuda
from tree_sitter import Language, Parser
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class CodeParser:
    def __init__(self):
        self.cpp_lang = Language(tscpp.language())
        self.cuda_lang = Language(tscuda.language())
        self.parsercpp = Parser(self.cpp_lang)
        self.parsercuda = Parser(self.cuda_lang)


    def detect_language(self, code_str):
        if code_str.startswith("#include <cuda.h>") or code_str.startswith("#include <cuda_runtime.h>"):
            lang = "cuda"
        else:
            lang = "cpp"
            
        if re.search(r'\b__global__\b', code_str):
            lang = "cuda"
        elif re.search(r'\b__device__\b', code_str):
            lang = "cuda"
        elif re.search(r'\b__host__\b', code_str):
            lang = "cuda"
        elif re.search(r'\b__shared__\b', code_str):
            lang = "cuda"
        return lang

    def parse_cpp(self, code_str):
        tree = self.parsercpp.parse(bytes(code_str, "utf8"))
        root_node = tree.root_node
        return root_node

    def parse_cuda(self, code_str):
        tree = self.parsercuda.parse(bytes(code_str, "utf8"))
        root_node = tree.root_node
        return root_node

    def parse(self, lang, code_str):
        if lang == "cpp":
            return self.parse_cpp(code_str)
        elif lang == "cuda":
            return self.parse_cuda(code_str)
        else:
            raise ValueError(f"Unsupported language: {lang}")

# AST Tokenizer wrapper
class ASTTokenizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    def walk_tree(self, node, code_bytes):
        tokens = [node.type]
        if node.child_count == 0:
            text = code_bytes[node.start_byte:node.end_byte].decode("utf8")
            return [text]
        else:
            for child in node.children:
                tokens += self.walk_tree(child, code_bytes)
        return tokens

    def tokenize_ast(self, root_node, code_str):
        code_bytes = bytes(code_str, "utf8")
        tokens = self.walk_tree(root_node, code_bytes)
        return tokens
        #return self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")