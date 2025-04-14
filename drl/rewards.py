# MCTS Immediate Rewards
def ir1_understand_serial(code_ast): ...
def ir2_find_bottlenecks(code_ast): ...
def ir3_analyze_dependencies(code_ast): ...
def ir4_choose_parallel_strategy(code_ast): ...
def ir5_implement_parallelism(code_ast): ...

# May need to normalize the reward
# to be between 0 and 1
def immediate_reward(state):
    return (
        ir1_understand_serial(state) +
        ir2_find_bottlenecks(state) +
        ir3_analyze_dependencies(state) +
        ir4_choose_parallel_strategy(state) +
        ir5_implement_parallelism(state)
    )

def ltr1_compile(code): ...
def ltr2_run(code): ...
def ltr3_performance(code): ...
def ltr4_verify(code, reference): ...

def long_term_reward(code, reference):
    return (
        ltr1_compile(code) +
        ltr2_run(code) +
        ltr3_performance(code) +
        ltr4_verify(code, reference)
    )

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def select(node): ...
def expand(node): ...
def simulate(node): return immediate_reward(node.state)
def backpropagate(node, value): ...

