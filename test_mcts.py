import os
import sys
import json
import time
import glob
import shutil
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(REPO_ROOT, "data", "deepscaler_first_200.json")
EXAMPLE_PATH = os.path.join(REPO_ROOT, "data", "example.json")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "output")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def list_output_dirs() -> set[str]:
    if not os.path.isdir(OUTPUT_ROOT):
        return set()
    return set([p for p in glob.glob(os.path.join(OUTPUT_ROOT, "*")) if os.path.isdir(p)])

def list_all_gv_files() -> set[str]:
    if not os.path.isdir(OUTPUT_ROOT):
        return set()
    return set(glob.glob(os.path.join(OUTPUT_ROOT, "**", "*.gv"), recursive=True))

def newest_by_mtime(paths: list[str]) -> str | None:
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))

def parse_gv_counts(gv_file: str) -> tuple[int, int]:
    # 返回 (green_count, yellow_count)
    green = yellow = 0
    try:
        with open(gv_file, "r", encoding="utf-8") as f:
            text = f.read()
        # 简单计数 fillcolor 标记
        green = text.count("fillcolor=lightgreen")
        yellow = text.count("fillcolor=lightyellow")
    except Exception:
        pass
    return green, yellow

def run_one_problem(i: int, sample: dict, run_log_dir: str) -> tuple[float, str]:
    """
    返回 (ratio, used_gv_path)
    """
    # 运行前快照
    pre_dirs = list_output_dirs()
    pre_gv = list_all_gv_files()

    # 覆盖 example.json 为单题输入
    tmp_inputs = [sample]
    write_json(EXAMPLE_PATH, tmp_inputs)

    # 调用 run.py
    cmd = [sys.executable, os.path.join(REPO_ROOT, "run.py"),
           "--model_type", "vllm",
           "--output_tree_vis",
           "--n_iters", "4",
           "--model_name_or_path","model/Qwen3-1.7B",
           "--gpu_memory_utilization","0.4"]
    log_path = os.path.join(run_log_dir, f"problem_{i:03d}.log")
    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write(f"[CMD] {' '.join(cmd)}\n")
        log_fp.flush()
        proc = subprocess.run(cmd, stdout=log_fp, stderr=subprocess.STDOUT, cwd=REPO_ROOT)

    # 运行后收集新增 .gv
    post_dirs = list_output_dirs()
    post_gv = list_all_gv_files()
    new_gv = list(post_gv - pre_gv)

    # 选择本题的 .gv 文件
    chosen_gv = None
    if new_gv:
        chosen_gv = newest_by_mtime(new_gv)
    else:
        # 未检测到新增时，从新增目录中找，若无则从所有目录找最新
        new_dirs = list(post_dirs - pre_dirs)
        if new_dirs:
            candidate_gv = glob.glob(os.path.join(newest_by_mtime(new_dirs), "**", "*.gv"), recursive=True)
            chosen_gv = newest_by_mtime(candidate_gv)
        if not chosen_gv:
            candidate_gv = list(post_gv)
            chosen_gv = newest_by_mtime(candidate_gv)

    # 统计比例
    ratio = 0.0
    if chosen_gv and os.path.isfile(chosen_gv):
        g, y = parse_gv_counts(chosen_gv)
        denom = g + y
        ratio = (g / denom) if denom > 0 else 0.0

    return ratio, chosen_gv or ""

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = os.path.join(OUTPUT_ROOT, f"test_mcts_{ts}")
    logs_dir = os.path.join(result_root, "logs")
    ensure_dir(logs_dir)

    # summary.log 路径
    summary_path = os.path.join(result_root, "summary.log")

    # 备份 example.json
    example_backup = None
    if os.path.exists(EXAMPLE_PATH):
        with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
            example_backup = f.read()

    # 读取已完成问题（断点续跑）
    done_indices = set()
    done_ratios = dict()
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Problem "):
                    parts = line.strip().split()
                    idx = int(parts[1].replace(":", ""))
                    ratio = float(parts[2].split("=")[1])
                    done_indices.add(idx)
                    done_ratios[idx] = ratio

    try:
        dataset = read_json(DATASET_PATH)
        if not isinstance(dataset, list) or len(dataset) == 0:
            print(f"[ERROR] Dataset invalid or empty: {DATASET_PATH}")
            sys.exit(1)

        ratios: list[float] = []
        per_item_result = []

        # 先统计已完成问题
        for idx in sorted(done_indices):
            ratios.append(done_ratios[idx])
            per_item_result.append({
                "index": idx,
                "ratio_correct_terminal": done_ratios[idx],
                "gv_path": "(already done)"
            })

        # 追加写 summary.log，每跑完一个问题就写一行
        with open(summary_path, "a", encoding="utf-8") as summary_fp:
            for i, sample in enumerate(dataset[:2]):
                if i in done_indices:
                    print(f"[SKIP] Problem {i+1} already done.")
                    continue
                print(f"[RUN] Problem {i+1}/{len(dataset)} ...")
                ratio, gv_path = run_one_problem(i, sample, logs_dir)
                ratios.append(ratio)
                per_item_result.append({
                    "index": i,
                    "ratio_correct_terminal": ratio,
                    "gv_path": os.path.relpath(gv_path, REPO_ROOT) if gv_path else ""
                })
                summary_fp.write(f"Problem {i:03d}: ratio={ratio:.6f} gv={os.path.relpath(gv_path, REPO_ROOT) if gv_path else ''}\n")
                summary_fp.flush()

        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0

        # 最后写平均值
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"\nAverage ratio over {len(ratios)} problems: {avg_ratio:.6f}\n")

        print(f"[DONE] Summary written to: {summary_path}")

    finally:
        # 恢复 example.json
        if example_backup is not None:
            with open(EXAMPLE_PATH, "w", encoding="utf-8") as f:
                f.write(example_backup)

if __name__ == "__main__":
    main()
