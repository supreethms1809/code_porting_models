# Set system path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import load_dataset, Dataset
from src.parse.parser import CodeParser, ASTTokenizer
from src.mcts.node import MCTS_search, SubsetMCTSNode, subset_mcts_search
from src.grpo.agent import codeGRPO
from src.reward_bridge.trace_buffer import TraceBuffer
from src.model_inference.deepseek_inference import modelInference
from src.parse.extract_analysis import AnalysisExtractor, AnalysisRewardScorer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load your HF dataset
dataset = Dataset.load_from_disk("/home/sureshm/ssuresh/code_porting_models/dataset/babeltower")
dataset = dataset.select(range(5))

parser = CodeParser()
grpo = codeGRPO()
buffer = TraceBuffer()
asttokenizer = ASTTokenizer(model_name)
default_action = "Translate and Parallelize code from cpp to CUDA"

for example in dataset:
    code_str = example["code"]

    ######### Phase: Read the code and generate the formatted AST ############
    lang = parser.detect_language(code_str)
    code_ast_root = parser.parse(lang, code_str)
    code_ast_walk = asttokenizer.format_ast_tree(code_ast_root, code_str.encode("utf-8"))

    ######### Phase: Analysis of the code ############
    analysis_result = modelInference().infer(code_ast_walk)
    extracted_fields = AnalysisExtractor().extract(analysis_result)
    analysis_score = AnalysisRewardScorer().score(extracted_fields)
    logger.info(f"Analysis Score: {analysis_score}")

    ######### Phase: Generate the trace using MCTS ############
    #trace_mcts = MCTS_search(code_str, extracted_fields, iterations=20)
    trace_mcts = subset_mcts_search(code_str, extracted_fields, iterations=20)
    # Generate the code with the best set of actions
    best_action = trace_mcts.selected_actions[-1] if trace_mcts.selected_actions else default_action
    translated_code = modelInference().infer_transform(code_str, best_action)
    logger.info(f"Translated Code: {translated_code.lower()}")

    ######### Phase: Evaluate the trace using GRPO ############
    #ltr_score = grpo.evaluate_trace(trace)
    #buffer.store(trace, ltr_score)

print(f"Finished processing {len(dataset)} examples.")