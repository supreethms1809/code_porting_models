from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_immediate_rewards import ComputeImmediateRewards
import hashlib

class MCTSNode:
    def __init__(self, code_str, parent=None, action=None):
        # Here code_str is the current state
        self.code_str = code_str
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_immediate_reward = 0.0
        self.uid = hashlib.md5(code_str.encode()).hexdigest()

    def exapand(self, action):
        transformed_code_str = llm_transform(self.code_str, action)
        for new_code in transformed_code_str:
            if not any(child.uid == hashlib.md5(new_code.encode()).hexdigest() for child in self.children):
                self.children.append(MCTSNode(new_code, parent=self))

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

def llm_transform(code_str, action):
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    template = """
    You are a code transformation model. Your task is to transform the given code snippet according to the specified action.
    The action is: {action}
    The code snippet is:
    {code_snippet}
    Please provide the transformed code snippet.
    """
    inputs = self.tokenizer(template.format(action=action, code_snippet=self.code_str), return_tensors="pt")
    outputs = self.model.generate(**inputs)
    transformed_code = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return transformed_code


def compute_immediate_rewards(code_str, transformed_code_str):
    compute_reward = ComputeImmediateRewards(code_str, transformed_code_str)
    # reward is normalized to [0, 1] here 
    return compute_reward.cuda_accuracy_score()

def mcts_search(root_code, llm, iterations=10):
    root = MCTSNode(root_code)
    for _ in range(iterations):
        node = root
        # Selection
        while node.children:
            node = node.best_child()
        # Expansion
        node.expand(llm)
        # Simulation (evaluate)
        if node.children:
            node = random.choice(node.children)
        reward = compute_immediate_rewards(node.code)
        node.backpropagate(reward)
    return root.best_child(c_param=0).code


