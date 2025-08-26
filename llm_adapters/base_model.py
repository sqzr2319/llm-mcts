from typing import List, Union


class BaseLLMModel():
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(
        self, 
        prompt:Union[str, List[dict], List[List[dict]], List[str]], 
        continue_last:bool = False,
        n_gen:int = 1,
        **kwargs
    ) -> tuple[str, dict]:
        raise NotImplementedError("This method should be implemented by subclasses.")