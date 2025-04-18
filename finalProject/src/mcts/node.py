from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_immediate_rewards import ComputeImmediateRewards
from src.parse.parser import CodeParser
from src.parse.parser import ASTTokenizer
from src.model_inference.deepseek_inference import modelInference
import hashlib
import random
import math
from typing import List, Optional, Dict, Any

TRANSFORMATION_ACTIONS = [
    "add '__global__' to the function",
    "replace_loop_index_with_thread_id_and_block_id",
    "wrap_loop_body_with_kernel",
    "insert_boundary_check",
    "insert_cuda_malloc",
    "insert_cuda_memcpy",
    "use_shared_memory",
    "insert_syncthreads",
    "flatten_nested_loops",
    "move_constants_to_device",
    "add_cuda_kernel_launch_host"
]

class MCTSNode:
    def __init__(self, code_str, parent=None, action=None):
        # Here code_str is the current state
        self.code_state = code_str

        # class specific variables
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_immediate_reward = 0.0
        self.untried_actions = []
        self.state_metadata = {}
        self.total_reward = 0.0
        self.uid = hashlib.md5(code_str.encode("utf-8")).hexdigest()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def bestChild(self, exploration_weight=1.41):
        def uct_score(child):
            if child.visits == 0:
                return float('inf')
            return (child.total_immediate_reward / child.visits) + exploration_weight * ((2 * (self.visits) ** 0.5) / (1 + child.visits))
        return max(self.children, key=uct_score)
    
    def backpropagate(self, reward: float):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def expand(self, action: str, new_code_str: str) -> 'MCTSNode':
        child_node = MCTSNode(
            code_str=new_code_str,
            parent=self,
            action=action
        )
        self.children.append(child_node)
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return child_node
    
    def __repr__(self):
        return f"<Node action={self.action}, visits={self.visits}, reward={self.total_reward:.2f}>"
    
def apply_action(code_str: str, action: str, extracted_fields: str):
    # Run it through the model to get the new code
    translated_code = modelInference().infer_transform(code_str, action)
    return translated_code

def MCTS_search(code_str: str, extracted_fields: str, iterations: int = 1000, exploration_weight: float = 1.41) -> MCTSNode:
    root = MCTSNode(code_str)
    root.untried_actions = TRANSFORMATION_ACTIONS.copy()
    for _ in range(iterations):
        node = root
        while not node.is_fully_expanded():
            if node.untried_actions:
                retry = True
                while retry:
                    action = random.choice(node.untried_actions)
                    translated_code = apply_action(node.code_state, action, extracted_fields)
                    if translated_code != "":
                        retry = False
                    else:
                        node.untried_actions.remove(action)
                node = node.expand(action, translated_code)
            else:
                node = node.bestChild(exploration_weight)

        # Create ast of new code
        code_ast_root = CodeParser().parse("cpp", code_str)
        new_code_ast_root = CodeParser().parse("cuda", node.code_state)

        immediate_reward = ComputeImmediateRewards(code_ast_root, new_code_ast_root).compute_all()
        node.backpropagate(immediate_reward)

    return root.bestChild(0)  # Return the best child of the root

class SubsetMCTSNode:
    def __init__(self, selected_actions: list, parent=None):
        self.selected_actions = selected_actions
        self.untried_actions = [a for a in TRANSFORMATION_ACTIONS if a not in selected_actions]
        self.children = []
        self.parent = parent
        self.visits = 0
        self.total_reward = 0.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        action = self.untried_actions.pop()
        new_subset = self.selected_actions + [action]
        child = SubsetMCTSNode(new_subset, parent=self)
        self.children.append(child)
        return child

    def best_child(self, exploration_weight=1.41):
        best_score = float("-inf")
        best_node = None
        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.total_reward / child.visits
            explore = (2 * math.log(self.visits + 1) / child.visits) ** 0.5
            score = exploit + exploration_weight * explore
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def __repr__(self):
        return f"<Node action={self.selected_actions}, visits={self.visits}, reward={self.total_reward:.2f}>"

    def __print__(self, level=0):
        print("\t" * level + repr(self))
        for child in self.children:
            child.__print__(level + 1)

def subset_mcts_search(code_str: str, extracted_fields: str, iterations: int = 1000, exploration_weight: float = 1.41) -> SubsetMCTSNode:
    root = SubsetMCTSNode([])

    for _ in range(iterations):
        node = root

        # Selection
        while not node.is_fully_expanded() and node.children:
            node = node.best_child(exploration_weight)

        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()

        # Apply the selected subset of actions as a combined instruction
        combined_prompt = "\n".join(node.selected_actions)
        translated_code = modelInference().infer_transform(code_str, combined_prompt)

        # Skip if generation fails
        if not translated_code.strip():
            continue

        # Parse ASTs
        cpp_ast = CodeParser().parse("cpp", code_str)
        cuda_ast = CodeParser().parse("cuda", translated_code)

        # Compute rewards
        rewards = ComputeImmediateRewards(cpp_ast, cuda_ast).compute_all()

        # Backpropagate
        node.backpropagate(rewards)

    return root.best_child(0) if root.children else root