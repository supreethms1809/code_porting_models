from src.parse.parser import CodeParser, ASTTokenizer

class ComputeImmediateRewards:
    def __init__(self, ast_cpp, ast_cuda):
        self.ast_cpp = ast_cpp
        self.ast_cuda = ast_cuda

        self.cuda_keywords = {
            "global", "device", "host", "shared", "constant", "managed", 
            "restrict", "noinline", "forceinline", "threadIdx", "blockIdx",
            "blockDim", "gridDim", "warpSize", "__syncthreads()",
            "__syncthreads_count()", "__syncthreads_and()",
            "__syncthreads_or()", "__syncwarp()", "__threadfence()",
            "__threadfence_block()", "__threadfence_system()",
            "atomicAdd", "atomicSub", "atomicExch", "atomicMin", "atomicMax",
            "atomicInc", "atomicDec", "atomicCAS", "atomicAnd", "atomicOr",
            "atomicXor"
        }

    def compute_all(self) -> dict:
        scores = {
            "IR1_loop_bounds": self.loop_bounds_preserved(),
            "IR2_index_usage": self.index_usage_correct(),
            "IR3_arithmetic": self.operations_match(),
            "IR4_dependency_respected": self.dependencies_respected(),
            "IR5_cuda_semantics": self.parallel_semantics_correct()
        }
        scores["total_immediate_reward"] = sum(scores.values()) / len(scores)
        return scores["total_immediate_reward"]

    def walk(self, node):
        tokens = [node]
        for child in node.children:
            tokens.extend(self.walk(child))
        return tokens

    def loop_bounds_preserved(self) -> float:
        cpp_loops = self.extract_nodes(self.ast_cpp, "for_statement")
        cuda_loops = self.extract_nodes_cuda(self.ast_cuda, "for_statement")
        if not cpp_loops or not cuda_loops:
            return 0.0
        cpp_init = cpp_loops[0].child_by_field_name("initializer")
        cuda_init = cuda_loops[0].child_by_field_name("initializer")
        return 1.0 if cpp_init and cuda_init and cpp_init.text == cuda_init.text else 0.0

    def index_usage_correct(self) -> float:
        cpp_indices = self.extract_nodes(self.ast_cpp, "array_subscript")
        cuda_indices = self.extract_nodes_cuda(self.ast_cuda, "array_subscript")
        return 1.0 if len(cpp_indices) == len(cuda_indices) else 0.0

    def operations_match(self) -> float:
        cpp_ops = {n.text for n in self.extract_nodes(self.ast_cpp, "binary_expression")}
        cuda_ops = {n.text for n in self.extract_nodes_cuda(self.ast_cuda, "binary_expression")}
        return 1.0 if cpp_ops.issubset(cuda_ops) else 0.0

    def dependencies_respected(self) -> float:
        writes = self.extract_nodes_cuda(self.ast_cuda, "assignment_expression")
        for node in writes:
            rhs = node.child_by_field_name("right")
            if rhs:
                rhs_text = rhs.text.decode("utf-8") if isinstance(rhs.text, bytes) else rhs.text
                if any(k in rhs_text for k in ("threadIdx", "blockIdx", "blockDim")):
                    return 0.0
        return 1.0

    def parallel_semantics_correct(self) -> float:
        return 1.0 #if any(kw in self.code_cuda for kw in self.cuda_keywords) else 0.0

    def extract_nodes(self, ast, node_type: str):
        return [node for node in self.walk(ast) if node.type == node_type]
    
    def extract_nodes_cuda(self, ast, node_type: str):
        return [node for node in self.walk(ast) if node.type == node_type]
