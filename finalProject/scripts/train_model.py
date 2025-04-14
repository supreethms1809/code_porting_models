# Set system path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import load_dataset, Dataset
from src.parse.parser import CodeParser, ASTTokenizer
from src.mcts.search import MCTSSearch
from src.grpo.agent import codeGRPO
from src.reward_bridge.trace_buffer import TraceBuffer
from src.transforms.serial_analysis import SerialAnalyzer
from src.transforms.bottleneck import BottleneckFinder
from src.transforms.dependency_graph import DependencyAnalyzer
from src.transforms.parallel_strategies import ParallelStrategySelector
from src.transforms.insert_parallelism import ParallelismInserter

# Load your HF dataset
dataset = Dataset.load_from_disk("/home/sureshm/ssuresh/code_porting_models/dataset/babeltower")
dataset = dataset.select(range(20))
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

parser = CodeParser()
mcts = MCTSSearch([
    SerialAnalyzer(),
    BottleneckFinder(),
    DependencyAnalyzer(),
    ParallelStrategySelector(),
    ParallelismInserter()
])
grpo = codeGRPO()
buffer = TraceBuffer()
asttokenizer = ASTTokenizer(model_name)

for example in dataset:
    code_str = example["code"]
    lang = parser.detect_language(code_str)
    parsed = parser.parse(lang, code_str)
    tokenized = asttokenizer.tokenize_ast(parsed, code_str)
    trace = mcts.run(parsed)
    ltr_score = grpo.evaluate_trace(trace)
    buffer.store(trace, ltr_score)

print(f"Finished processing {len(dataset)} examples.")