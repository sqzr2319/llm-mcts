import json, os
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    # experiment config
    parser.add_argument("--entry", type=str, default='mcts', choices=['mcts', 'infer'])
    parser.add_argument("--save_dir", type=str, default='./output')
    parser.add_argument("--exp_name", type=str, default='default_exp')
    parser.add_argument("--input_file", type=str, default='data/example.jsonl')
    
    # model config
    parser.add_argument("--model_type", type=str, default='vllm', choices=['vllm', 'debug'])
    parser.add_argument("--model_name_or_path", type=str, default="/data/NAS/llm_model_weights/Qwen3-1.7B")
    
    # search config
    parser.add_argument("--action_prompt_config", type=str, default='prompt/action_qwen_math.json')
    
    # mcts config
    parser.add_argument("--w_exp", type=float, default=1.0)
    parser.add_argument("--n_actions", type=int, default=4)
    parser.add_argument("--depth_limit", type=int, default=5)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--cum_reward", type=str, default='sum')
    parser.add_argument("--calc_q", type=str, default='mean')
    
    # mcts save config
    parser.add_argument("--output_tree_vis", action='store_true', default=False)
    
    # vllm config
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    
    args = parser.parse_args()
    
    # append time to exp_name
    from datetime import datetime
    args.exp_name = f"{args.exp_name}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # save all args to a json file
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # copy all code to save_dir/code_backup and ignore by .gitignore
    os.system(f"rsync -av --exclude-from='.gitignore' ./ {args.save_dir}/code_backup/")
    
    
    if args.entry == 'mcts':
        from mcts import _main
        _main(args)
    else:
        raise NotImplementedError(f"Entry point '{args.entry}' is not implemented.")
    