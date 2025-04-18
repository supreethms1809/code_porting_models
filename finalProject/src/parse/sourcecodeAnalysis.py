################# Deprecated #################
# Delete this file later after testing

import os
from src.model_inference.deepseek_inference import modelInference

class SourceCodeAnalyzer:
    def __init__(self):
        self.ir1 = 0
        self.ir2 = 0
        self.ir3 = 0
        self.ir4 = 0
        self.ir5 = 0
        self.final_analysis_score = 0

    # Loop analysis score based on the number of loops
    def loop_analysis_score(self, loop_analysis):
        return loop_analysis.get("score", 0)

    def index_analysis_score(self, index_analysis):
        return index_analysis.get("score", 0)

    def operation_analysis_score(self, operation_analysis):
        return operation_analysis.get("score", 0)

    def dependency_analysis_score(self, dependency_analysis):
        return dependency_analysis.get("score", 0)

    def parallel_analysis_score(self, parallel_analysis):
        return parallel_analysis.get("score", 0)

    def parseAnalysis(self, analysis_results):
        loop_analysis = analysis_results.get("loop_analysis", {})
        index_analysis = analysis_results.get("index_analysis", {})
        operation_analysis = analysis_results.get("operation_analysis", {})
        dependency_analysis = analysis_results.get("dependency_analysis", {})
        parallel_analysis = analysis_results.get("parallel_analysis", {})

        ir1 = self.loop_analysis_score(loop_analysis)
        ir2 = self.index_analysis_score(index_analysis)
        ir3 = self.operation_analysis_score(operation_analysis)
        ir4 = self.dependency_analysis_score(dependency_analysis)
        ir5 = self.parallel_analysis_score(parallel_analysis)

        # Combine the scores to get the final analysis score
        final_analysis_score = (ir1 + ir2 + ir3 + ir4 + ir5) / 5

        return final_analysis_score


    def llm_analysis_code(self, code_ast):
        # Use the model to analyze the code
        analysis_results = modelInference().infer(code_ast)
        analysis_score = self.parseAnalysis(analysis_results)
        return analysis_score