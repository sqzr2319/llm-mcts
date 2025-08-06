from llm_adapters.base_model import BaseLLMModel

class DebugModel(BaseLLMModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Additional initialization if needed

    def generate(self, prompt: str, continue_last: bool = False, n_gen:int=1, **kwargs) -> tuple[str, dict]:
        # Implement the generation logic for debugging purposes
        if isinstance(prompt, str):
            prompt = [[{"role": "user", "content": prompt}],]
        elif isinstance(prompt, list) and isinstance(prompt[0], str):
            prompt = [[{"role": "user", "content": p} for p in prompt],]
        elif isinstance(prompt, list) and isinstance(prompt[0], dict):
            prompt = [prompt]
        return [[f"debug response action_{i}_{j}_section_1\n\ndebug response action_{i}_{j}_section_2\n\ndebug response\nend {i}_{j}###final answer:$$\n\\boxed{{57}}\n$$" \
            for j in range(n_gen)] \
            for i in range(len(prompt))], \
            [[{"debug_info": "This is a debug response"}for __ in range(n_gen)] for _ in range(len(prompt))]