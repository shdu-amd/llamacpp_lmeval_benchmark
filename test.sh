# # Examples: drive lmms-eval against a running llama.cpp server.
# # See run_model_bench.py for all flags. The CLI is unchanged from the old
# # lm-eval client; only the underlying harness switched to lmms-eval.

# # python3 run_model_bench.py --base-url http://127.0.0.1:8080 --model-name Qwen3.5-0.8b-Q4_1  --num-concurrent 4 --benchmark mmlu_pro --limit 1 --checkpoint-every 1 --random-seed 0


# # python3 run_model_bench.py --base-url http://127.0.0.1:8080 --model-name gemma-4-E4B-it-Q4_1 --num-concurrent 1 --benchmark mmlu_pro --limit 1 --checkpoint-every 1 --random-seed 0


# # python3 run_model_bench.py --base-url http://127.0.0.1:8080 --model-name Qwen3.5-4b-Q4_1 --num-concurrent 16 --benchmark mmlu_pro --limit 4 --checkpoint-every 1 --random-seed 0


# # python3 run_model_bench.py --base-url http://127.0.0.1:8080 --model-name gemma-4-12b-it-Q4_1 --num-concurrent 64 --benchmark mmlu_pro --limit 6 --checkpoint-every 2 --random-seed 0


# # python3 run_model_bench.py --base-url http://127.0.0.1:8080 --model-name gemma-4-12b-it-Q4_1 --num-concurrent 48 --benchmark mmlu_pro  --checkpoint-every 10 --random-seed 0


# # python3 run_model_bench.py --base-url http://127.0.0.1:8080 --model-name gemma-4-12b-it-Q4_1 --num-concurrent 24 --benchmark mmlu_pro  --limit 300 --checkpoint-every 25 --random-seed 0

# # Use mmlu_pro_std (task_overrides/) for the canonical custom-extract-only filter.
# python3 run_model_bench.py --base-url http://127.0.0.1:8080 --model-name gemma-4-12b-it-Q4_1 --num-concurrent 8 --benchmark mmlu_pro_std  --checkpoint-every 1 --random-seed 0
# # python3 run_model_bench.py --base-url http://127.0.0.1:8080 --model-name Qwen3.5-0.8b-Q4_1  --num-concurrent 8 --benchmark mmlu_pro --limit 3 --checkpoint-every 1 --random-seed 0


# todo, check the yaml to see if it is correct




#python3 run_modelscope_bench.py --datasets mmlu_pro --api-url http://127.0.0.1:8080 --eval-batch-size 24 --limit 10 --model-name Qwen3.5-0.8B --max-tokens 4096

#python3 run_modelscope_bench.py --datasets mmmu_pro --api-url http://127.0.0.1:8080 --eval-batch-size 24 --limit 10 --model-name Qwen3.5-0.8B --max-tokens 4096

python3 run_modelscope_bench.py --datasets bfcl_v4 --api-url http://127.0.0.1:8080 --eval-batch-size 24 --limit 2 --model-name Qwen3.5-0.8B --max-tokens 4096