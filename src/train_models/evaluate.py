# File to hold the evaluate class for model evaluation
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import Dataset
from codebleu import calc_codebleu
import code_bert_score
import re
import evaluate
import json
import os
from Levenshtein import distance as levenshtein_distance 

class Evaluate():
    def __init__(self):
        # self.model = model
        # self.tokenizer = tokenizer
        self.bleu_metric = None
        self.codebleu_metric = None
        self.parableu_metric = None
        self.cuda_keywords = ["global", "device", "host", "shared", "constant", "managed", 
								"restrict", "noinline", "forceinline", "threadIdx", "blockIdx",
								"blockDim", "gridDim", "warpSize", "__syncthreads()",
								"__syncthreads_count()", "__syncthreads_and()",
								"__syncthreads_or()", "__syncwarp()", "__threadfence()",
								"__threadfence_block()", "__threadfence_system()",
								"atomicAdd", "atomicSub", "atomicExch", "atomicMin", "atomicMax",
								"atomicInc", "atomicDec", "atomicCAS", "atomicAnd", "atomicOr",
								"atomicXor"]

    def eval(self, file_path):
        # Placeholder for evaluation logic
        # This should include loading the model, running inference, and calculating metrics
        # Load the inference output 
        with open(file_path, "r") as f:
            data = json.load(f)
            predictions = [item['kernel_code'] for item in data]
            references = [item['reference'] for item in data]

        # Calculate metrics
        bleu = self.bleu_score(predictions, references)
        codebleu = self.codebleu_score(predictions, references)
        parableu = self.parableu_score(predictions, references, codebleu)
        code_BERTscore = self.BERTcodescore(predictions, references)

        #codebleu = 0.0
        #parableu = 0.0 
        #code_BERTscore = 0.0

        return {
            "bleu": bleu,
            "codebleu": codebleu,
            "parableu": parableu,
            "code_BERTscore": code_BERTscore
        }

    def bleu_score(self, predictions, references):
        # Placeholder for BLEU score calculation
        # This should include the actual BLEU score calculation logic
        bleu = evaluate.load('bleu')
        self.bleu_metric = bleu.compute(
            predictions=predictions,
            references=references
        )
        return self.bleu_metric


    def codebleu_score(self, predictions, references):
        # Placeholder for CodeBLEU score calculation
        # This should include the actual CodeBLEU score calculation logic
        codebleu = calc_codebleu(
            predictions=predictions,
            references=references,
            lang = "cpp",
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        return codebleu
    
    def BERTcodescore(self, predictions, references):
        # Placeholder for BERTScore calculation
        # This should include the actual BERTScore calculation logic
        P, R, F1, F3 = code_bert_score.score(
            cands=predictions,
            refs=references,
            lang = "cpp"
        )
        return P.mean(), R.mean(), F1.mean(), F3.mean()

    def cuda_similarity(self, ref: str, pred: str) -> float:
        ref_kw = {kw for kw in self.cuda_keywords if kw in ref}
        pred_kw = {kw for kw in self.cuda_keywords if kw in pred}
        if not ref_kw: return 1.0
        return len(ref_kw & pred_kw) / len(ref_kw)

    def extract_loop_structure(self, code: str) -> list:
        code = ''.join(code)
        return re.findall(r'\bfor\b|\bwhile\b', code)

    def loop_similarity(self, ref: str, pred: str) -> float:
        ref_loops = self.extract_loop_structure(ref)
        pred_loops = self.extract_loop_structure(pred)
        max_len = max(len(ref_loops), len(pred_loops))
        if max_len == 0: return 1.0
        dist = levenshtein_distance(ref_loops, pred_loops)
        return 1 - dist / max_len

    def parallel_semantics_similarity(self, ref: str, pred: str) -> float:
        ref = ''.join(ref)
        seq_loops = len(re.findall(r'for\s*\(.*;.*;.*\)', ref))
        parallel_threads = pred.count('threadIdx')
        if seq_loops == 0: return 1.0
        matched = min(seq_loops, parallel_threads)
        return matched / seq_loops

    def parableu_score(self, predictions, references, codebleu):
        # Placeholder for ParaBLEU score calculation
        # This should include the actual ParaBLEU score calculation logic
        codebleu = codebleu["codebleu"]
        sim_cuda = self.cuda_similarity(references, predictions)
        sim_loops = self.loop_similarity(references, predictions)
        sim_parallel = self.parallel_semantics_similarity(references, predictions)
        print(f"CUDA similarity: {sim_cuda}, Loop similarity: {sim_loops}, Parallel similarity: {sim_parallel}")
        parableu = codebleu * sim_cuda * sim_loops * sim_parallel
        return parableu

    def plot_metrics(self, metrics):
        # Placeholder for plotting metrics
        # This should include the actual plotting logic
        pass
