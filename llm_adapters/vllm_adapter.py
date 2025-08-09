from llm_adapters.base_model import BaseLLMModel
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from typing import TypedDict, List, Dict, Any, Union, Optional

class VLLMAdapter(BaseLLMModel):
    def __init__(
            self, 
            model_name: str,
            **kwargs):
        super().__init__(model_name)
        self.llm = LLM(model=model_name, **kwargs)
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _encode_message_without_last_gen(self, messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        eos_token = self.tokenizer.eos_token
        index = text.rfind(eos_token)
        if index != -1:
            text = text[:index]
        return text
        
    def _generate(self, messages: Union[list[dict], list[list[dict]]], continue_last: bool = False, n_gen:int = 1, **kwargs) -> tuple[str, dict]:
        if not isinstance(messages[0], list):
            messages = [messages]
        if continue_last:
            messages = [self._encode_message_without_last_gen(msg) for msg in messages]
        sample_params = SamplingParams(
            max_tokens = 8192,
            n = n_gen
        )
        outputs = self.llm.generate(messages, sampling_params=sample_params)
        output_texts = [[output.outputs[i].text for i in range(len(output.outputs))] for output in outputs]
        output_infos = [{
            'logprobs': [output.outputs[i].logprobs for i in range(len(output.outputs))],
            'token_ids': [output.outputs[i].token_ids for i in range(len(output.outputs))],
            'prompt': output.prompt,
        } for output in outputs]
        # print("[DEBUG] generated outputs:", output_texts)
        return output_texts, output_infos
    
    def generate(
        self, 
        prompt: Union[str, List[Dict[str, Any]]], 
        continue_last: bool = False,
        n_gen: int = 1,
        **kwargs
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        if isinstance(prompt, str):
            prompt = [[{"role": "user", "content": prompt}],]
        elif isinstance(prompt, list) and isinstance(prompt[0], str):
            prompt = [[{"role": "user", "content": p} for p in prompt],]
        elif isinstance(prompt, list) and isinstance(prompt[0], dict):
            prompt = [prompt]
        assert isinstance(prompt, list) and isinstance(prompt[0], list) and isinstance(prompt[0][0], dict), "Prompt must be a list of lists of dictionaries."
        return self._generate(prompt, continue_last=continue_last, n_gen=n_gen, **kwargs)
        

if __name__ == "__main__":
    model_name = "/data/NAS/llm_model_weights/Qwen3-1.7B"
    adapter = VLLMAdapter(model_name, gpu_memory_utilization=0.8)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The cap"}
    ]
    response, info = adapter._generate(messages, continue_last=True)
    print("Response:", response[0])
    print("Info:", info[0])
    print("Token IDs:", info[0]['token_ids'])
