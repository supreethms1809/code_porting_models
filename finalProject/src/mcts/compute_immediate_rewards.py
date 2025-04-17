class ComputeImmediateRewards():
    def __init__(self, ast_cpp, ast_cuda):
        self.ast_cpp = ast_cpp
        self.ast_cuda = ast_cuda
        self.cuda_keywords = ["global", "device", "host", "shared", "constant", "managed", 
                        "restrict", "noinline", "forceinline", "threadIdx", "blockIdx",
                        "blockDim", "gridDim", "warpSize", "__syncthreads()",
                        "__syncthreads_count()", "__syncthreads_and()",
                        "__syncthreads_or()", "__syncwarp()", "__threadfence()",
                        "__threadfence_block()", "__threadfence_system()",
                        "atomicAdd", "atomicSub", "atomicExch", "atomicMin", "atomicMax",
                        "atomicInc", "atomicDec", "atomicCAS", "atomicAnd", "atomicOr",
                        "atomicXor"]
        self.walk = ASTTokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").walk_tree
        self.parser = CodeParser()

    def loop_bounds_preserved(self, ast_cpp, ast_cuda):
        cpp_loops = self.extract_for_loops(ast_cpp)
        cuda_loops = self.extract_thread_loops(ast_cuda)

        if not cpp_loops or not cuda_loops:
            return False

        cpp_range = self.get_loop_range(cpp_loops[0])
        cuda_range = self.get_thread_range(cuda_loops[0])

        return cpp_range == cuda_range
    
    def index_usage_correct(self, ast_cpp, ast_cuda):
        cpp_indices = self.extract_array_accesses(ast_cpp)
        cuda_indices = self.extract_array_accesses(ast_cuda)

        for c, g in zip(cpp_indices, cuda_indices):
            if not self.is_index_equivalent(c, g):
                return False
        return True

    def operations_match(self, ast_cpp, ast_cuda):
        cpp_exprs = self.extract_arithmetic_expressions(ast_cpp)
        cuda_exprs = self.extract_arithmetic_expressions(ast_cuda)

        return all(expr in cuda_exprs for expr in cpp_exprs)

    def dependencies_respected(self, ast_cpp, ast_cuda):
        writes = extract_writes(ast_cuda)
        for var, index_expr in writes:
            if self.depends_on_other_threads(index_expr):
                return False
        return True

    def parallel_semantics_correct(self, ast_cuda):
        return any(kw in ast_cuda.text for kw in self.cuda_keywords)

    def extract_for_loops(self, ast):
        return [n for n in self.walk(ast) if n.type == 'for_statement']

    def get_loop_range(self, loop_node):
        return (loop_node.init, loop_node.condition, loop_node.increment)

    def extract_array_accesses(self, ast):
        return [node for node in ast.walk() if node.type == 'array_subscript']

    def extract_arithmetic_expressions(self, ast):
        return [node.text for node in ast.walk() if node.type == 'binary_operator']

    def extract_writes(self, ast):
        return [(node.lhs, node.rhs) for node in ast.walk() if node.type == 'assignment']

    def cuda_accuracy_score(self):
        score = 0
        if self.loop_bounds_preserved(self.ast_cpp, self.ast_cuda):
            score += 1
        if self.index_usage_correct(self.ast_cpp, self.ast_cuda):
            score += 1
        if self.operations_match(self.ast_cpp, self.ast_cuda):
            score += 1
        if self.dependencies_respected(self.ast_cpp, self.ast_cuda):
            score += 1
        if self.parallel_semantics_correct(self.ast_cuda):
            score += 1
        normalized_score = score / 5.0
        return normalized_score
