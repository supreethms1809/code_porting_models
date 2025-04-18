from typing import Dict

class AnalysisExtractor:
    def __init__(self):
        self.fields = {
            "loop_semantics": {
                "embarrassingly_parallel": None,
                "has_data_dependencies": None,
                "fusable": None,
                "splittable": None
            },
            "loop_dependencies": {
                "loop_carried": None
            },
            "parallel_region": {
                "compute_vs_memory_bound": None,
                "has_conditionals": None
            }
        }

    def extract(self, report: str) -> Dict:
        r = self.fields.copy()
        text = report.lower()

        # Embarrassingly parallel
        if any(x in text for x in [
            "embarrassingly parallel",
            "each iteration is independent",
            "no inter-iteration dependency",
            "loop can be parallelized"
        ]):
            r["loop_semantics"]["embarrassingly_parallel"] = "Yes"
        elif "not embarrassingly parallel" in text:
            r["loop_semantics"]["embarrassingly_parallel"] = "No"

        # Data dependencies
        if any(x in text for x in [
            "no data dependency",
            "independent iterations",
            "no shared variables",
            "no dependencies between iterations"
        ]):
            r["loop_semantics"]["has_data_dependencies"] = "No"
        elif any(x in text for x in [
            "has data dependency",
            "dependent on previous iteration",
            "shared variable modified"
        ]):
            r["loop_semantics"]["has_data_dependencies"] = "Yes"

        # Loop-carried dependencies
        if "loop-carried dependency" in text or "carried dependency" in text:
            r["loop_dependencies"]["loop_carried"] = "Yes"
        elif "no loop-carried dependency" in text:
            r["loop_dependencies"]["loop_carried"] = "No"

        # Fusable
        if "loops are fusable" in text or "loops can be fused" in text:
            r["loop_semantics"]["fusable"] = "Yes"
        elif "not fusable" in text or "cannot be fused" in text:
            r["loop_semantics"]["fusable"] = "No"

        # Splittable
        if "loops are splittable" in text or "loops can be split" in text:
            r["loop_semantics"]["splittable"] = "Yes"
        elif "not splittable" in text or "cannot be split" in text:
            r["loop_semantics"]["splittable"] = "No"

        # Compute/memory bound
        if "compute-bound" in text:
            r["parallel_region"]["compute_vs_memory_bound"] = "Compute-bound"
        elif "memory-bound" in text:
            r["parallel_region"]["compute_vs_memory_bound"] = "Memory-bound"

        # Conditional exits
        if any(x in text for x in [
            "conditional exits", "break", "return", "continue"
        ]):
            r["parallel_region"]["has_conditionals"] = "Yes"
        elif "no conditional exits" in text:
            r["parallel_region"]["has_conditionals"] = "No"

        return r

class AnalysisRewardScorer:
    def __init__(self):
        self.weights = {
            "embarrassingly_parallel": {"Yes": 1.0, "No": -0.5},
            "has_data_dependencies": {"No": 1.0, "Yes": -0.5},
            "loop_carried": {"No": 1.0, "Yes": -1.0},
            "fusable": {"Yes": 0.5, "No": 0.0},
            "splittable": {"Yes": 0.5, "No": 0.0},
            "compute_vs_memory_bound": {"Compute-bound": 0.5, "Memory-bound": 0.2},
            "has_conditionals": {"Yes": -0.5, "No": 0.2}
        }

    def score(self, extracted_fields: Dict) -> Dict[str, float]:
        reward_dict = {
            "IR1_embarrassingly_parallel": 0.0,
            "IR2_has_data_dependencies": 0.0,
            "IR3_loop_carried": 0.0,
            "IR4_fusable": 0.0,
            "IR5_splittable": 0.0,
            "IR6_compute_vs_memory_bound": 0.0,
            "IR7_has_conditionals": 0.0
        }

        sem = extracted_fields.get("loop_semantics", {})
        dep = extracted_fields.get("loop_dependencies", {})
        par = extracted_fields.get("parallel_region", {})

        reward_dict["IR1_embarrassingly_parallel"] = self.weights["embarrassingly_parallel"].get(
            sem.get("embarrassingly_parallel"), 0)

        reward_dict["IR2_has_data_dependencies"] = self.weights["has_data_dependencies"].get(
            sem.get("has_data_dependencies"), 0)

        reward_dict["IR3_loop_carried"] = self.weights["loop_carried"].get(
            dep.get("loop_carried"), 0)

        reward_dict["IR4_fusable"] = self.weights["fusable"].get(
            sem.get("fusable"), 0)

        reward_dict["IR5_splittable"] = self.weights["splittable"].get(
            sem.get("splittable"), 0)

        reward_dict["IR6_compute_vs_memory_bound"] = self.weights["compute_vs_memory_bound"].get(
            par.get("compute_vs_memory_bound"), 0)

        reward_dict["IR7_has_conditionals"] = self.weights["has_conditionals"].get(
            par.get("has_conditionals"), 0)

        reward_dict["total_analysis_reward"] = sum(reward_dict.values())
        return reward_dict