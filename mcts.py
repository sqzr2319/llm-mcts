import json, time, os
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable, Union
import itertools
from abc import ABC
from abc import ABC
from collections import defaultdict

import numpy as np
from tqdm import trange
try:
    from utils.profiler import profile, patch_methods
except Exception:
    # profiling is optional
    def profile(*args, **kwargs):
        class _N:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        return _N()
    def patch_methods(*args, **kwargs):
        return None

from search_config import MathConfig
from world_model import MathProblemEnv, MathPromptDict, MathAction, MathState, action_to_json
State = MathState
Action = MathAction


class MCTSNode():
    id_iter = itertools.count()
    search_step_iter = 0

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()
        cls.search_step_iter = 0
    
    @classmethod
    def increment_search_step(cls):
        cls.search_step_iter += 1

    def __init__(self, state: Optional[State], action: Optional[Action], parent: "Optional[MCTSNode]" = None,
                 fast_reward: float = 0., fast_reward_details=None,
                 is_terminal: bool = False, calc_q: Callable[[list[float]], float] = np.mean):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(MCTSNode.id_iter)
        self.create_step = MCTSNode.search_step_iter
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[MCTSNode]]' = None
        self.calc_q = calc_q
        self.expanded = False  # whether the node has been expanded
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float:
        if self.state is None:
            return self.fast_reward
        else:
            return self.calc_q(self.cum_rewards)
    
    @property
    def full_info_dict(self) -> dict:
        return {
            'id': self.id,
            'create_step': self.create_step,
            'state': self.state,
            'action': self.action,
            'fast_reward': self.fast_reward,
            'is_terminal': self.is_terminal,
            'depth': self.depth,
            'cum_rewards': list(self.cum_rewards),
            'Q': self.Q,
            'parent_id': self.parent.id if self.parent else None,
        }
    
    @property
    def simple_info_dict(self) -> dict:
        return {
            'id': self.id,
            'create_step': self.create_step,
            'action': action_to_json(self.action),
            'cum_rewards': list(self.cum_rewards),
            'is_terminal': self.is_terminal,
            'parent_id': self.parent.id if self.parent else None,
        }


class MCTSResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: Optional[tuple[list[State], list[Action]]]
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    tree_state_after_each_iter: list[MCTSNode] = None
    aggregated_result: Optional[Hashable] = None


class MCTSAggregation(ABC):
    def __init__(self, retrieve_answer: Callable[[State], Hashable],
                 weight_policy: str = 'edge'):
        assert weight_policy in ['edge', 'edge_inverse_depth', 'uniform']
        self.retrieve_answer = retrieve_answer
        self.weight_policy = weight_policy

    def __call__(self, tree_state: MCTSNode) -> Optional[Hashable]:
        answer_dict = defaultdict(lambda: 0)

        def visit(cur: MCTSNode):
            if cur.state is None:
                return []
            if cur.is_terminal:
                answer = self.retrieve_answer(cur.state)
                if answer is None:
                    print("MCTSAggregation: no answer retrieved.")
                    return []
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / cur.depth
                elif self.weight_policy == 'uniform':
                    answer_dict[answer] += 1.0
                return [(answer, cur.depth)]
            depth_list = defaultdict(list)
            cur_list = []
            for child in cur.children:
                cur_list.extend(child_info := visit(child))
                for answer, depth in child_info:
                    depth_list[answer].append(depth)
            for answer, depths in depth_list.items():
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / np.mean(depths)
            return cur_list

        visit(tree_state)

        if len(answer_dict) == 0:
            return None
        return max(answer_dict, key=lambda answer: answer_dict[answer])

class MCTS():
    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 n_iters: int = 10,
                 cum_reward: Callable[[list[float]], float] = sum,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: Union[str, Callable[[list[float]], int]] = 'max',
                 output_strategy: str = 'max_reward',
                 uct_with_fast_reward: bool = True,
                 fast_simulate: bool = True,
                 aggregator: Optional[MCTSAggregation] = None,
                 disable_tqdm: bool = True,
                 node_visualizer: Callable[[MCTSNode], dict] = lambda x: x.__dict__):
        """
        MCTS algorithm

        :param output_trace_in_each_iter: whether to output the trace of the chosen trajectory in each iteration ; the trace is *deepcopy*-ed
                                          will also output *tree_state_after_each_iter*, which is the *deepcopy*-ed root
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param output_strategy: the way to output the result. The nodes are not *deepcopy*-ed, so the information is after all iterations
                                Options: 'max_reward': dfs on the final tree to find a trajectory with max reward using :param cum_reward:
                                         'follow_max': starting from root, choose the maximum reward child at each step. May output a non-terminal node if dead end
                                         'max_visit': the terminal node with maximum number of visits
                                         'max_iter': the trajectory with a terminal node and max reward among those in each iteration
                                         'last_iter': the last trajectory. May output a non-terminal node if the last iteration leads to a dead end
                                         'last_terminal_iter': the last trajectory with a terminal node
                                Outputs *None* if no trajectory with terminal node but required
        :param uct_with_fast_reward: if True, use fast_reward instead of reward for unvisited children in UCT
                                     Otherwise, visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                             simulate_strategy)
        assert output_strategy in ['max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter',
                                   'last_terminal_iter']
        self.output_strategy = output_strategy
        self.fast_simulate = fast_simulate
        self.uct_with_fast_reward = uct_with_fast_reward
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        MCTSNode.increment_search_step()
        with profile("mcts.iterate", meta={"fast": self.fast_simulate}):
            return self._iterate_fast_simulate(node) if self.fast_simulate else self._iterate(node)
        
    
    def _iterate_fast_simulate(self, node: MCTSNode) -> list[list[MCTSNode]]:
        with profile("mcts.select"):
            path = self._select(node)
        print("Selected path: root\n\t" + '\n\t'.join([n.action.content for n in path[1:]]))
        if not self._is_terminal_with_depth_limit(path[-1]):
            with profile("mcts.fast_simulate"):
                new_path_list = self._fast_simulate(path)
            with profile("mcts.back_propagate_fast"):
                cum_reward = self._back_propagate_fast_simulate(path, new_path_list)
            
            for new_path in new_path_list:
                cum_reward = self.cum_reward([node.reward for node in path]+[node.reward for node in new_path])
                if self.output_strategy == 'max_iter' and cum_reward > self._output_cum_reward:
                    self._output_cum_reward = cum_reward
                    self._output_iter = path + new_path
                elif self.output_strategy == 'last_iter' or self.output_strategy == 'last_terminal_iter':
                    self._output_cum_reward = cum_reward
                    self._output_iter = path + new_path
        
        return path
    
    def _iterate(self, node: MCTSNode) -> list[MCTSNode]:
        with profile("mcts.select"):
            path = self._select(node)
        if not self._is_terminal_with_depth_limit(path[-1]):
            with profile("mcts.expand"):
                self._expand(path[-1])
            with profile("mcts.simulate"):
                self._simulate(path)
        with profile("mcts.back_propagate"):
            cum_reward = self._back_propagate(path)
        if self.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > self._output_cum_reward:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_iter':
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        return path

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []
        while True:
            path.append(node)
            # FIXME(dengchaoyi): 这份代码可能有别的地方用 children是不是空来判断有没有expand过，需要改掉
            if not node.expanded or self._is_terminal_with_depth_limit(node):
                return path
            node = self._uct_select(node)

    def _uct(self, node: MCTSNode) -> float:
        return node.Q + self.w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards)))

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
            return max(node.children, key=self._uct)
        else:
            unvisited_children = filter(lambda x: x.state is None, node.children)
            return max(unvisited_children, key=lambda x: x.fast_reward)

    def _expand(self, node: MCTSNode):
        if node.state is None:
            node.state, aux = self.world_model.step(node.parent.state, node.action)
            # reward is calculated after the state is updated, so that the
            # information can be cached and passed from the world model
            # to the reward function with **aux without repetitive computation
            # node.reward, node.reward_details = self.search_config. \
            #     reward(node.parent.state, node.action, **node.fast_reward_details, **aux)
            # We should compute reward when the state is created
            node.is_terminal = self.world_model.is_terminal(node.state)
        node.expanded = True

        if node.is_terminal:
            return

        children = []
        actions = self.search_config.get_actions(node.state)
        for action in actions:
            fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, action)
            child = MCTSNode(state=None, action=action, parent=node,
                             fast_reward=fast_reward, fast_reward_details=fast_reward_details, calc_q=self.calc_q)
            children.append(child)

        node.children = children

    def _simulate(self, path: list[MCTSNode]):
        node = path[-1]
        while True:
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            fast_rewards = [child.fast_reward for child in node.children]
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)
    
    
    def _fast_simulate(self, path: list[MCTSNode]) -> list[list[MCTSNode]]:
        node = path[-1]
        if node.state is None:
            # same as expand
            node.state, aux = self.world_model.step(node.parent.state, node.action)
            node.is_terminal = self.world_model.is_terminal(node.state)
        node.expanded = True
        if node.is_terminal:
            return 
        print("Fast simulate from node:", node.id, "with state:", node.state)
        actions_list = self.search_config.fast_simulate(node.state, n_actions=self.search_config.n_actions)
        new_path_list = []
        print(actions_list)
        for actions in actions_list:
            new_path_list.append([])
            # 先把第一个action挂到node下
            fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, actions[0]) # 实际上只有最后一个action reward非0
            child = MCTSNode(state=None, action=actions[0], parent=node,
                             fast_reward=fast_reward, fast_reward_details=fast_reward_details, calc_q=self.calc_q,
                             is_terminal=actions[0].is_end)
            if node.children is None:
                node.children = []
            node.children.append(child)
            new_path_list[-1].append(child)
            cur_node = child
            for action in actions[1:]:
                # 后面的action依次挂成一个链
                fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, action) 
                child = MCTSNode(state=None, action=action, parent=cur_node,
                                 fast_reward=fast_reward, fast_reward_details=fast_reward_details, calc_q=self.calc_q,
                                 is_terminal=action.is_end)
                if cur_node.children is None:
                    cur_node.children = []
                cur_node.children.append(child)
                new_path_list[-1].append(child)
                cur_node = child
            cur_node.is_terminal = True
        return new_path_list

    def _back_propagate_fast_simulate(self, path: list[MCTSNode], sample_path: list[list[MCTSNode]]):
        # 这里计算方法和内源代码对齐，需要讨论是否有必要修改
        # 假设expand的节点产生了n个sample路径，会先更新每个sample路径上的node，每个node visit次数+1
        # 然后将所有sample路径的cum_reward取平均，作为采样的reward，再从expand节点向上更新，路上每个节点仍是+1 visit次数（而不是+n）
        total_sample_reward = []
        for sample in sample_path:
            rewards = []
            cum_reward = -math.inf
            for node in reversed(sample):
                rewards.append(node.reward)
                cum_reward = self.cum_reward(rewards[::-1])
                node.cum_rewards.append(cum_reward)
            total_sample_reward.append(cum_reward)
        mean_sample_reward = np.mean(total_sample_reward)
        print("backpropagate fast simulate with mean sample reward:", mean_sample_reward, " full list:", total_sample_reward)
        rewards = [mean_sample_reward]
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(mean_sample_reward)
        return cum_reward
                

    def _back_propagate(self, path: list[MCTSNode]):
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(cum_reward)
        return cum_reward

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.state is None and len(path) > 1:
            cur.state, aux = self.world_model.step(path[-2].state, cur.action)
            cur.is_terminal = self.world_model.is_terminal(cur.state)
            
        if cur.is_terminal:
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])
    
    def _dfs_collect_tree_info(self, use_simple_info: bool = True) -> list[dict]:
        def visit(node: MCTSNode):
            if node.children is None:
                return [node.simple_info_dict if use_simple_info else node.full_info_dict]
            return [node.simple_info_dict if use_simple_info else node.full_info_dict] + \
                   list(itertools.chain.from_iterable(visit(child) for child in node.children))

        return visit(self.root)

    def search(self):
        self._output_cum_reward = -math.inf
        self._output_iter = None
        with profile("world.init_state"):
            self.root = MCTSNode(state=self.world_model.init_state(), action=None, parent=None, calc_q=self.calc_q)
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        for i in trange(self.n_iters, disable=self.disable_tqdm, desc='MCTS iteration', leave=False):
            with profile("mcts.iteration", meta={"iter": i}):
                path = self.iterate(self.root)
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

        if self.output_strategy == 'follow_max':
            self._output_iter = []
            cur = self.root
            while True:
                self._output_iter.append(cur)
                if cur.is_terminal:
                    break
                visited_children = [x for x in cur.children if x.state is not None]
                if len(visited_children) == 0:
                    break
                cur = max(visited_children, key=lambda x: x.reward)
            self._output_cum_reward = self.cum_reward([node.reward for node in self._output_iter[1::-1]])
        if self.output_strategy == 'max_reward':
            self._output_cum_reward, self._output_iter = self._dfs_max_reward([self.root])
            if self._output_cum_reward == -math.inf:
                self._output_iter = None
    

    def __call__(self,
                 world_model: MathProblemEnv,
                 search_config: MathConfig,
                 log_file: Optional[str] = None,
                 **kwargs) -> MCTSResult:
        MCTSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config

        self.search()

        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = [node.state for node in self._output_iter], [node.action for node in self._output_iter[1:]]
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0] for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None
        result = MCTSResult(terminal_state=terminal_state,
                            cum_reward=self._output_cum_reward,
                            trace=trace,
                            trace_of_nodes=self._output_iter,
                            tree_state=self.root,
                            trace_in_each_iter=trace_in_each_iter,
                            tree_state_after_each_iter=tree_state_after_each_iter)
        if self.aggregator is not None:
            result = MCTSResult(
                terminal_state=result.terminal_state,
                cum_reward=result.cum_reward,
                trace=result.trace,
                trace_of_nodes=result.trace_of_nodes,
                tree_state=result.tree_state,
                trace_in_each_iter=result.trace_in_each_iter,
                tree_state_after_each_iter=result.tree_state_after_each_iter,
                aggregated_result=self.aggregator(result.tree_state),
            )
        return result
    
def parse_cum_reward(cum_reward: str) -> Callable[[list[float]], float]:
    if cum_reward == 'sum':
        return sum
    elif cum_reward == 'mean':
        return np.mean
    elif cum_reward == 'max':
        return max
    else:
        raise ValueError(f"Unknown cum_reward: {cum_reward}")

def parse_calc_q(calc_q: str) -> Callable[[list[float]], float]:
    if calc_q == 'mean':
        return np.mean
    elif calc_q == 'max':
        return max
    elif calc_q == 'sum':
        return sum
    else:
        raise ValueError(f"Unknown calc_q: {calc_q}")
    
def _main(configs):
    from llm_adapters import get_model_class_by_name
    
    
    from world_model import MathProblemEnv
    from search_config import MathConfig
    
    # build base model
    with profile("load.base_model", meta={"type": configs.model_type}):
        base_model = get_model_class_by_name(configs.model_type)(
            configs.model_name_or_path,
            gpu_memory_utilization=configs.gpu_memory_utilization,)
    print("[MCTS] Base model loaded:", base_model)
    
    # build world model
    with profile("build.world_model"):
        world_model = MathProblemEnv()
    print("[MCTS] World model built:", world_model)
    
    # build search config
    # prompt = MathPromptDict(
    #     system_prompt="You are a helpful assistant for solving math problems.",
    #     question_prefix="Solve the following math problem: ",
    #     question="What is (2213 + 23)/2?",
    #     question_suffix="Let's solve it step by step.",
    #     gt='1118'
    # )
    with profile("build.search_config"):
        search_config = MathConfig(
            base_model=base_model,
            n_actions=configs.n_actions,
        )
    # search_config.update_example(prompt=prompt)
    print("[MCTS] Search config built:", search_config)
    
    print("[MCTS] Start to load input prompts from:", configs.input_file)
    with profile("load.prompts", meta={"path": configs.input_file}):
        if configs.input_file.endswith('.jsonl'):
            with open(configs.input_file, 'r') as f:
                prompts = [json.loads(line) for line in f]
        elif configs.input_file.endswith('.json'):
            with open(configs.input_file, 'r') as f:
                prompts = json.load(f)
    print("[MCTS] Input prompts loaded, total len:", len(prompts))
    
    action_prompt_config = None
    if configs.action_prompt_config:
        print("[MCTS] Load action prompt config from:", configs.action_prompt_config)
        with profile("load.action_prompt_config", meta={"path": configs.action_prompt_config}):
            with open(configs.action_prompt_config, 'r') as f:
                action_prompt_config = json.load(f)
        print("[MCTS] Action prompt config loaded:", action_prompt_config)
    assert action_prompt_config is not None
    
    for prompt_index, prompt in enumerate(prompts):
        start_time = time.time()
        print("[MCTS] Update search config with prompt:", prompt)
        if isinstance(prompt, dict):
            prompt = MathPromptDict(**prompt)
        elif isinstance(prompt, str):
            prompt = MathPromptDict(question=prompt)
        else:
            raise ValueError(f"Unknown prompt format: {type(prompt)}")
        prompt = MathPromptDict(
            system_prompt=prompt.get('system_prompt', action_prompt_config["system_prompt"]),
            question_prefix=prompt.get('question_prefix', action_prompt_config["question_prefix"]),
            question=prompt['question'],
            question_suffix=prompt.get('question_suffix', action_prompt_config["question_suffix"]),
            gt=prompt.get('gt', None)
        )
        with profile("search_config.update_example"):
            search_config.update_example(prompt=prompt)
    
        # build MCTS
        mcts = MCTS(
            output_trace_in_each_iter=True,
            w_exp=configs.w_exp,
            n_iters=configs.n_iters,
            fast_simulate=True,
            cum_reward=parse_cum_reward(configs.cum_reward),
            calc_q=parse_calc_q(configs.calc_q),
        )

        print("[MCTS] MCTS start searching...")
        # Patch base_model.generate to measure LLM latency per call
        try:
            from llm_adapters.base_model import BaseLLMModel
            patch_methods([
                (BaseLLMModel, "generate", "llm.generate"),
                (type(search_config.base_model), "generate", "llm.generate.concrete"),
            ])
        except Exception:
            pass
        with profile("mcts.call", meta={"prompt_index": prompt_index}):
            results = mcts(world_model=world_model, search_config=search_config)
        infer_time = time.time() - start_time
        print("[MCTS] MCTS search finished, time taken:", infer_time, "seconds")
        max_reward, max_path = mcts._dfs_max_reward([mcts.root])
        max_response = '\n\n'.join([action.content for action in max_path[-1].state])
        print("[MCTS] Search max_reward:", max_reward, "max_response:", max_response)
        with open(os.path.join(configs.save_dir, f"infer_results.jsonl"), 'a', encoding='utf-8') as f:
            json.dump({
                'prompt': prompt,
                'infer_result': max_response,
                'gt': prompt['gt'],
                'max_reward': max_reward,
                'infer_time': infer_time,
            }, f, ensure_ascii=False)
        mcts_info = mcts._dfs_collect_tree_info(use_simple_info=True)
        with open(os.path.join(configs.save_dir, f"mcts_info.jsonl"), 'a', encoding='utf-8') as f:
            json.dump({
                'prompt': prompt,
                'tree_info': mcts_info,
                'infer_time': infer_time,
            }, f, ensure_ascii=False)
        if configs.output_tree_vis:
            from utils import visualize_tree
            visualize_tree(mcts_info, 
                           os.path.join(configs.save_dir, f"tree_vis/mcts_tree_{prompt_index}.gv"),
                           gt=prompt.get('gt', "")),