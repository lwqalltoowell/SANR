import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent


def script_module(script: str) -> str:
    """Convert a repository-relative Python entry point to a module name."""
    script_path = Path(script)
    if (
        script_path.is_absolute()
        or script_path.suffix != ".py"
        or not all(part.isidentifier() for part in script_path.with_suffix("").parts)
    ):
        raise ValueError(f"script must be a repository-relative Python module: {script}")
    return ".".join(script_path.with_suffix("").parts)


def load_config(path: Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是 mapping")
    return config


def load_training_parser(script_path: Path) -> argparse.ArgumentParser:
    if not script_path.exists():
        raise ValueError(f"训练脚本不存在: {script_path}")
    module_name = f"_experiment_script_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载训练脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_parser"):
        raise ValueError(f"训练脚本缺少 build_parser(): {script_path}")
    parser = module.build_parser()
    if not isinstance(parser, argparse.ArgumentParser):
        raise ValueError(f"{script_path} 的 build_parser() 必须返回 ArgumentParser")
    return parser


def parser_defaults(parser: argparse.ArgumentParser) -> dict:
    return {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help"
    }


def validate_config(config: dict, config_path: Path) -> tuple[str, dict, dict]:
    required_top_level = {"default_task", "default_seeds", "tasks", "methods"}
    missing = required_top_level - config.keys()
    if missing:
        raise ValueError(f"配置缺少字段: {sorted(missing)}")

    default_task = config["default_task"]
    default_seeds = config["default_seeds"]
    tasks = config["tasks"]
    methods = config["methods"]
    if not isinstance(default_task, str):
        raise ValueError("default_task 必须是字符串")
    if not isinstance(default_seeds, list) or not default_seeds:
        raise ValueError("default_seeds 必须是非空列表")
    if not all(isinstance(seed, int) for seed in default_seeds):
        raise ValueError("default_seeds 必须全部是整数")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("tasks 必须是非空 mapping")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("methods 必须是非空 mapping")
    if default_task not in tasks:
        raise ValueError(f"default_task 不存在: {default_task}")

    normalized_tasks = {}
    for task_name, task in tasks.items():
        if not isinstance(task_name, str) or not isinstance(task, dict):
            raise ValueError("tasks 中的每一项都必须是 name -> mapping")
        if (
            not set(task).issubset({"script", "seeds", "params"})
            or "script" not in task
        ):
            raise ValueError(f"任务 {task_name} 只能包含 script、可选的 seeds 和 params")
        if not isinstance(task["script"], str):
            raise ValueError(f"任务 {task_name} 的 script 必须是字符串")
        script_module(task["script"])
        seeds = task.get("seeds", default_seeds)
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"任务 {task_name} 的 seeds 必须是非空列表")
        if not all(isinstance(seed, int) for seed in seeds):
            raise ValueError(f"任务 {task_name} 的 seeds 必须全部是整数")
        params = task.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"任务 {task_name} 的 params 必须是 mapping")
        if not all(isinstance(key, str) for key in params):
            raise ValueError(f"任务 {task_name} 的参数名必须是字符串")
        normalized_tasks[task_name] = {
            "script": task["script"],
            "script_path": (REPO_ROOT / task["script"]).resolve(),
            "seeds": seeds,
            "params": params,
        }

    normalized_methods = {}
    for method_name, method in methods.items():
        if not isinstance(method_name, str) or not isinstance(method, dict):
            raise ValueError("methods 中的每一项都必须是 name -> mapping")
        if set(method) != {"group", "params"}:
            raise ValueError(f"方法 {method_name} 只能包含 group 和 params")
        if not isinstance(method["group"], str):
            raise ValueError(f"方法 {method_name} 的 group 必须是字符串")
        if not isinstance(method["params"], dict):
            raise ValueError(f"方法 {method_name} 的 params 必须是 mapping")
        if not all(isinstance(key, str) for key in method["params"]):
            raise ValueError(f"方法 {method_name} 的参数名必须是字符串")
        normalized_methods[method_name] = {
            "group": method["group"],
            "params": method["params"],
        }
    return default_task, normalized_tasks, normalized_methods


def validate_params(task: dict, methods: dict, valid_params: set[str], task_name: str):
    task_params = set(task["params"])
    unknown_task_params = task_params - valid_params
    if unknown_task_params:
        raise ValueError(
            f"任务 {task_name} 含有未知参数: {sorted(unknown_task_params)}"
        )

    reserved_params = task_params & {"seed", "group_name"}
    if reserved_params:
        raise ValueError(
            f"任务 {task_name} 的 params 不能覆盖调度参数: {sorted(reserved_params)}"
        )

    for method_name, method in methods.items():
        method_params = set(method["params"])
        unknown = method_params - valid_params
        if unknown:
            raise ValueError(
                f"任务 {task_name} 的方法 {method_name} 含有未知参数: {sorted(unknown)}"
            )
        conflicts = task_params & method_params
        if conflicts:
            raise ValueError(
                f"任务 {task_name} 与方法 {method_name} 的参数冲突: {sorted(conflicts)}"
            )


def build_cmd(script: str, task_params: dict, method: dict, seed: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        script_module(script),
        "--seed",
        str(seed),
        "--group_name",
        method["group"],
    ]
    for key, value in task_params.items():
        cmd.extend([f"--{key}", str(value)])
    for key, value in method["params"].items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def run_fingerprint(
    task_name: str,
    script: str,
    task_params: dict,
    method_name: str,
    method: dict,
    seed: int,
) -> str:
    payload = {
        "task": task_name,
        "script": script,
        "task_params": task_params,
        "method": method_name,
        "group": method["group"],
        "seed": seed,
        "params": method["params"],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def load_progress(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    completed = data.get("completed", [])
    if not isinstance(completed, list):
        raise ValueError(f"进度文件格式错误: {path}")
    if not all(isinstance(item, dict) and "id" in item for item in completed):
        raise ValueError(f"进度文件格式已过期，请删除后重新运行: {path}")
    return {item["id"]: item for item in completed}


def save_progress(path: Path, completed: dict[str, dict]):
    with open(path, "w") as f:
        json.dump({"completed": sorted(completed.values(), key=lambda item: item["id"])}, f, indent=2)


def run_single(cmd: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=False, text=True)
        if result.returncode != 0:
            return False, f"exit code {result.returncode}"
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="可塑性实验调度器",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "experiments.yaml"),
        help="实验配置文件路径",
    )
    parser.add_argument("--task", type=str, help="要运行的任务名；默认使用 YAML 中的 default_task")
    parser.add_argument(
        "--parallel",
        nargs="?",
        const=0,
        type=int,
        default=1,
        help="并行运行的 seed 数。不传值则自动设为 CPU 核心数。默认 1（串行）。",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="跳过同一配置指纹下已经成功完成的实验项",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印要运行的命令，不实际执行",
    )
    parser.add_argument(
        "--method",
        action="append",
        type=str,
        help="只运行指定方法（可重复使用，如 --method SSR --method Redo）",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="只运行指定的种子（可重复使用，如 --seed 42 --seed 171）",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=180,
        help="每个实验之间的冷却时间（秒），仅在串行模式下生效 (默认: 180)",
    )
    args = parser.parse_args()

    # 限制并行数
    max_parallel = args.parallel
    if max_parallel <= 0:
        import os
        max_parallel = os.cpu_count() or 4

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    default_task, tasks, methods = validate_config(config, config_path)
    task_name = args.task or default_task
    if task_name not in tasks:
        raise ValueError(f"未知任务: {task_name}")
    task = tasks[task_name]

    training_parser = load_training_parser(task["script_path"])
    defaults = parser_defaults(training_parser)
    validate_params(task, methods, set(defaults), task_name)

    if args.method:
        unknown_methods = sorted(set(args.method) - set(methods))
        if unknown_methods:
            raise ValueError(f"未知方法: {unknown_methods}")
        selected_methods = {name: methods[name] for name in args.method}
    else:
        selected_methods = methods

    selected_seeds = args.seed if args.seed else task["seeds"]
    queue = []
    for method_name, method in selected_methods.items():
        for seed in selected_seeds:
            fingerprint = run_fingerprint(
                task_name,
                task["script"],
                task["params"],
                method_name,
                method,
                seed,
            )
            queue.append(
                {
                    "id": fingerprint,
                    "task": task_name,
                    "script": task["script"],
                    "task_params": task["params"],
                    "method": method_name,
                    "group": method["group"],
                    "params": method["params"],
                    "seed": seed,
                }
            )

    progress_file = config_path.parent / f"experiment_progress.{config_path.stem}.json"
    completed = load_progress(progress_file) if args.skip_completed else {}
    skipped = [item for item in queue if item["id"] in completed]
    if args.skip_completed:
        queue = [item for item in queue if item["id"] not in completed]

    if not queue:
        print("✅ 所有实验已完成，无需运行。")
        return

    # ── 打印实验计划 ──
    print(f"\n{'='*60}")
    print(f"任务: {task_name}")
    print(f"训练脚本: {task['script']}")
    print(f"计划: {len(selected_methods)} 种方法 × {len(selected_seeds)} 种子 = {len(queue)} 次待运行")
    print(f"并行度: {'串行' if max_parallel == 1 else f'并行 {max_parallel} 个'}")
    if skipped:
        print(f"已跳过: {len(skipped)} 个已完成项")
    print(f"{'='*60}\n")

    configured_param_names = sorted(
        set(task["params"])
        | {key for method in selected_methods.values() for key in method["params"]}
    )
    if configured_param_names:
        print("当前脚本基线:")
        for name in configured_param_names:
            print(f"  {name} = {defaults[name]}")
        print()

    if task["params"]:
        print("任务参数:")
        for key, value in task["params"].items():
            print(f"  {key} = {value}")
        print()

    for method_name, method in selected_methods.items():
        print(f"  [{method_name}] group={method['group']}")
        for k, v in method["params"].items():
            print(f"    {k} = {v}")

    print(f"\n种子: {selected_seeds}")
    print(f"剩余 {len(queue)} 次运行\n")

    if args.dry_run:
        print(f"\n{'─'*60}")
        print("DRY RUN — 将执行以下命令:\n")
        for item in queue:
            cmd = build_cmd(
                item["script"], item["task_params"], item, item["seed"]
            )
            print(f"  {' '.join(cmd)}")
        print(f"\n总计: {len(queue)} 条命令")
        return

    # ── 开始调度 ──
    start_time = time.time()
    if not args.skip_completed and progress_file.exists():
        progress_file.unlink()

    total = len(queue)
    success_count = 0
    failed_runs = []

    if max_parallel == 1:
        # ── 串行 ──
        for idx, item in enumerate(queue, 1):
            cmd = build_cmd(
                item["script"], item["task_params"], item, item["seed"]
            )
            exp_key = f"{item['task']}/{item['method']}/seed{item['seed']}"

            print(f"\n[{idx}/{total}] 运行 {exp_key}")
            print(f"  命令: {' '.join(cmd)}")

            success, err = run_single(cmd)
            if success:
                print(f"  ✅ {exp_key} 完成")
                completed[item["id"]] = item
                save_progress(progress_file, completed)
                success_count += 1
            else:
                print(f"  ❌ {exp_key} 失败: {err}")
                failed_runs.append((exp_key, err))

            if idx < total and args.cooldown > 0:
                print(f"  冷却 {args.cooldown}s ...")
                time.sleep(args.cooldown)

    else:
        # ── 并行 ──
        def task(item):
            cmd = build_cmd(
                item["script"], item["task_params"], item, item["seed"]
            )
            exp_key = f"{item['task']}/{item['method']}/seed{item['seed']}"
            success, err = run_single(cmd)
            return item, exp_key, success, err

        pool = ThreadPoolExecutor(max_workers=max_parallel)
        futures = []
        for item in queue:
            futures.append(pool.submit(task, item))

        for future in as_completed(futures):
            item, exp_key, success, err = future.result()
            if success:
                print(f"  ✅ {exp_key} 完成")
                completed[item["id"]] = item
                save_progress(progress_file, completed)
                success_count += 1
            else:
                print(f"  ❌ {exp_key} 失败: {err}")
                failed_runs.append((exp_key, err))

        pool.shutdown(wait=True)

    # ── 汇总 ──
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"运行完成! 耗时: {elapsed/3600:.1f}h")
    print(f"跳过: {len(skipped)}")
    print(f"本轮成功: {success_count}/{total}")

    if failed_runs:
        print(f"失败: {len(failed_runs)}/{total}")
        for exp_key, err in failed_runs:
            print(f"  - {exp_key}: {err}")
    else:
        print("🎉 全部成功!")

    # 全部完成则删除进度文件（干净收尾）
    if not failed_runs and progress_file.exists():
        progress_file.unlink()


if __name__ == "__main__":
    main()
