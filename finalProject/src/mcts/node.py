from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_immediate_rewards import ComputeImmediateRewards
from src.parse.parser import CodeParser
from src.parse.parser import ASTTokenizer
import hashlib
import random

class MCTSNode:
    def __init__(self, code_str, code_ast_root, code_ast_walk, parent=None, action=None, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        # Here code_str is the current state
        self.code_str = code_str
        self.code_ast_root = code_ast_root
        self.code_ast_walk = code_ast_walk
        self.transformed_code_str = []
        self.transformed_ast_tree = []
        self.transformed_ast_walk = []

        # class specific variables
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_immediate_reward = 0.0
        self.uid = hashlib.md5(code_str.encode("utf-8")).hexdigest()

        # Parse the transformed code string to get the AST
        self.parser = CodeParser()
        self.walk = ASTTokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").walk_tree

    def expand(self, code_str, code_ast_walk, action, n_samples=4):
        # Need structured output, add langchain here, apply action in sequence vs all at once
        transformed_code_list = []
        while len(transformed_code_list) < n_samples:
            transformed_code_str = llm_transform(code_ast_walk, action, self.model, self.tokenizer)
            transformed_code_list.append(transformed_code_str)
        for new_code in transformed_code_str:
            self.transformed_code_str.append(new_code)
            new_ast_tree = self.parser.parse(self.parser.detect_language(new_code), new_code)
            self.transformed_ast_tree.append(new_ast_tree)
            new_ast_walk = self.walk(new_ast_tree, new_code.encode("utf-8"))
            self.transformed_ast_walk.append(new_ast_walk)
            if not any(child.uid == hashlib.md5(new_code.encode()).hexdigest() for child in self.children):
                self.children.append(MCTSNode(new_code, new_ast_tree, new_ast_walk, parent=self, action=action, model=self.model, tokenizer=self.tokenizer))

    def best_child(self, exploration_weight=1.41):
        def uct_score(child):
            if child.visits == 0:
                return float('inf')
            return (child.total_immediate_reward/child.visits) + exploration_weight * ((2 * (self.visits) ** 0.5) / (1 + child.visits))
        return max(self.children, key=uct_score)

    def backpropagate(self, reward):
        self.visits += 1
        self.total_immediate_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

def llm_transform(code_ast_walk, action, model, tokenizer):
    template = """
    You are a code transformation model. Your task is to transform the given code snippet according to the specified action.
    The action is: {action}
    The code snippet is:
    {code_snippet}
    Please provide the transformed code snippet. Start with think but only provide the code snippet and don't add any other text.
    """
    inputs = tokenizer(template.format(action=action, code_snippet=code_ast_walk), return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=4098, do_sample=True, temperature=0.6, top_p=0.9, top_k=50)
    transformed_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return transformed_code


def compute_immediate_rewards(code_ast_walk, transformed_ast_walk):
    compute_reward = ComputeImmediateRewards(code_ast_walk, transformed_ast_walk)
    # reward is normalized to [0, 1] here 
    return compute_reward.cuda_accuracy_score()

def mcts_search(code_str, code_ast_root, code_ast_walk, action, model, tokenizer, iterations=10):
    mcts_node = MCTSNode(code_str, code_ast_root, code_ast_walk, parent=None, model=model, tokenizer=tokenizer)
    for _ in range(iterations):
        node = mcts_node
        # Selection
        while node.children:
            node = node.best_child()
        # Expansion
        node.expand(code_str, code_ast_walk, action)
        # Simulation (evaluate)
        if node.children:
            node = random.choice(node.children)

        reward = compute_immediate_rewards(node.code_ast_root, node.transformed_ast_root)
        node.backpropagate(reward)
    return mcts_node.best_child(c_param=0).code


