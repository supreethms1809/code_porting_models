from datasets import load_dataset
from src.parser.parser import ClangParser
from src.mcts.search import MCTSSearch
from src.grpo.agent import DummyGRPO
#from src.reward_bridge.trace_buffer import TraceBuffer
from src.transforms.serial_analysis import SerialAnalyzer
from src.transforms.bottleneck import BottleneckFinder
from src.transforms.dependency_graph import DependencyAnalyzer
from src.transforms.parallel_strategies import ParallelStrategySelector
from src.transforms.insert_parallelism import ParallelismInserter

# Load your HF dataset
dataset = load_dataset("json", data_files="data/code_data.json")["train"]

parser = ClangParser()
mcts = MCTSSearch([
    SerialAnalyzer(),
    BottleneckFinder(),
    DependencyAnalyzer(),
    ParallelStrategySelector(),
    ParallelismInserter()
])
grpo = DummyGRPO()
buffer = TraceBuffer()

for example in dataset:
    code_str = example["code"]
    parsed = parser.parse(code_str)
    trace = mcts.run(parsed)
    ltr_score = grpo.evaluate_trace(trace)
    buffer.store(trace, ltr_score)

print(f"Finished processing {len(dataset)} examples.")