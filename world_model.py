import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
import numpy as np


class MathAction(NamedTuple):
    content:str
    logprob: np.ndarray
    is_end: bool

def action_to_json(action: MathAction) -> dict:
    if action is None:
        return {}
    return {
        'content': action.content,
        'logprob': action.logprob.tolist(),
        'is_end': action.is_end
    }


MathState = list[MathAction]


class MathPromptDict(TypedDict):
    system_prompt: str
    question_prefix: str
    question: str
    question_suffix: str
    gt:str
    
    
class MathProblemEnv():
    """
    Math problem environment for interactive question answering.
    """

    def __init__(self) -> None:
        super().__init__()

    def update_example(self, example: str, prompt: MathPromptDict = None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example

    def init_state(self) -> list:
        return []

    def step(self, state: MathState, action: MathAction) -> tuple[MathState, dict]:
        state = state.copy()
        state.append(action)
        return state, {}

    def is_terminal(self, state: MathState) -> bool:
        
        return state[-1].is_end if state else False
