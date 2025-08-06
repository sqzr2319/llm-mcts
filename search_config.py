import io
import re
from typing import TypedDict, Optional
import numpy as np

from world_model import MathPromptDict, MathAction, MathState
from utils import is_equiv, basic_response_filter
from llm_adapters.base_model import BaseLLMModel

from typing import List


class MathConfig():
    def __init__(self,
                 base_model:BaseLLMModel,
                 n_actions:int = 4,
                 depth_limit:int =5,
                 force_terminating_on_depth_limit=True,
                 force_overall_prompt_on_overall_question=True,
                 force_overall_question_on_overall_prompt=True) -> None:
        self.base_model = base_model
        self.n_actions = n_actions
        self.depth_limit = depth_limit
        
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt
        

    def update_example(self, prompt: MathPromptDict = None) -> None:
        assert prompt is not None
        self.prompt = prompt

    def get_actions(self, state) -> list[MathAction]:
        pass
    
    def fast_reward(self, state: MathState, action: MathAction) -> tuple[float, dict]:
        return self.reward(state, action)
    
    def _format_prompt(self) -> List[dict]:
        prompt = self.prompt
        if prompt is None:
            raise ValueError("Prompt is not set.")
        formatted_prompt  = [
            {"role": "system", "content": prompt['system_prompt']},
            {"role": "user", "content": prompt['question_prefix'] + prompt['question'] + prompt['question_suffix']},
        ]
        return formatted_prompt
    def _split_response(self, text:str, mode:str = "\n\n") -> list[MathAction]:
        # 暂时没处理logprob
        text = basic_response_filter(text)
        actions = [MathAction(content=action.strip(), logprob=np.array([]), is_end=False) for action in text.split(mode) if action.strip(' \n')]
        actions[-1] = MathAction(content=actions[-1].content, logprob=actions[-1].logprob, is_end=True)  # Mark the last action as end
        return actions
    
    def _full_rollout(self, state:MathState, n_actions:int = None) -> List[List[MathAction]]:
        prompt_messages = self._format_prompt()
        prompt_messages.append({"role": "user", "content": state[-1].content if state else ""})
        
        if n_actions is None:
            n_actions = 1
        prompt_messages_list = [prompt_messages]
        response_text_list, info_dicts = self.base_model.generate(prompt_messages_list, continue_last=True, n_gen=n_actions)
        print("[DEBUG] response_text_list:", response_text_list)
        actions_list = [self._split_response(response_text) for response_text in response_text_list[0]]
        return actions_list # List[List[MathAction]]
        
    
    def fast_simulate(self, state: MathState, n_actions:int = None)->List[List[MathAction]]:
        if n_actions is None:
            n_actions = self.n_actions
        
        all_action_list = self._full_rollout(state, n_actions=n_actions)
            
        return all_action_list
        

    def calculate_reward(self, state: MathState, action: MathAction):
        return 1.0 if is_equiv(action.content, self.prompt['gt'], verbose=False) else -1.0

    def reward(self, state: MathState, action: MathAction) -> tuple[float, dict]:
        if action.is_end:
            return self.calculate_reward(state, action), {}
        else:
            return 0.0, {}
