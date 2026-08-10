import argparse
import os

from evalscope import TaskConfig, run_task
from evalscope.constants import (
    DEFAULT_DATASET_CACHE_DIR,
    DEFAULT_WORK_DIR,
    EvalBackend,
    EvalType,
    HubType,
    JudgeStrategy,
    ModelTask,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evalscope benchmark against a llama.cpp server.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mmlu_pro"],
        help="One or more dataset names to evaluate.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8080",
        help="Base URL of the OpenAI-compatible API (llama.cpp server).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=24,
        help="Number of parallel requests; match llama.cpp parallelism.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples per category to run (default: all).",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen3.5-0.8B",
        help="Model name reported to the API.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4 * 1024,
        help="Maximum number of tokens to generate per request.",
    )
    parser.add_argument(
        "--output-dir-prefix",
        type=str,
        default="",
        help="prefix for outuput directory"
    )
    return parser.parse_args()


args = parse_args()



api_url=args.api_url  # url from llamacpp
use_cache = None # do not use cache
eval_batch_size = args.eval_batch_size # match to llamacpp number of paralle work
eval_type="openai_api"
limit= args.limit  # for now, run it all # number of datasample to run per category
dataset_hub= HubType.HUGGINGFACE  # for speed, use from huggingface
model_name=args.model_name
model_task= ModelTask.TEXT_GENERATION
enable_progress_tracker=True # write a progress.json to eanable some realtime logging
data_sample_seed=42
max_tokens=args.max_tokens
analysis_report=True
datasets=args.datasets
results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

if args.output_dir_prefix:
    work_dir = os.path.join(results_root, args.output_dir_prefix)
else:
    work_dir = results_root
os.makedirs(work_dir, exist_ok=True)
dataset_args={
    # nothing for the dataset 
    'mmmu_pro':{
        "dataset_id":"MMMU/MMMU_Pro" # from huggingface id
    }
}


generation_config = {
    "temperature": 0.0,   # greedy — the actual lever on the server path
    "do_sample": False,
    "seed": data_sample_seed,           # tie-breaking determinism; llama.cpp honors it
    "max_tokens": max_tokens,
    "timeout": 900,       # generous; 15s will kill long generations
    "retries":30,
    "stream": True,       # required for TTFT in perf metrics
}

taskcfg = TaskConfig(
    model=model_name,
    api_url=api_url,
    eval_batch_size=eval_batch_size,
    eval_type=eval_type,
    dataset_hub=dataset_hub,
    model_task=model_task,
    data_sample_seed=data_sample_seed,
    analysis_report=analysis_report,
    datasets=datasets,
    dataset_args=dataset_args,
    use_cache=use_cache,
    enable_progress_tracker=enable_progress_tracker,
    limit=limit,
    work_dir=work_dir,
    generation_config=generation_config
)
run_task(task_cfg=taskcfg)
