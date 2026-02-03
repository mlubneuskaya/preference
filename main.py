# main.py

import os

from src.training.sft import train_sft
from src.training.dpo import train_dpo
from src.utils.evaluate import evaluate_file
from src.utils.inference import generate_responses
from src.utils.formatting_functions import format_gsm8k, format_dpo_data
from transformers import AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B"

SFT_OUTPUT_DIR = "./output/sft_model"
DPO_FULL_OUTPUT_DIR = "./output/dpo_full_model"
DPO_1K_OUTPUT_DIR = "./output/dpo_1k_model" 
LOGS_DIR = "./output/logs"

RESULTS_BASE = "./output/results/base_results.jsonl"
RESULTS_SFT = "./output/results/sft_results.jsonl"
RESULTS_DPO_FULL = "./output/results/dpo_full_results.jsonl"
RESULTS_DPO_1K = "./output/results/dpo_1k_results.jsonl"

EVAL_BASE = "./output/results/base_summary.csv"
EVAL_SFT = "./output/results/sft_summary.csv"
EVAL_DPO_FULL = "./output/results/dpo_full_summary.csv"
EVAL_DPO_1K = "./output/results/dpo_1k_summary.csv"

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_INFER = 64
GRAD_ACCUM = 4
EPOCHS = 1


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_BASE), exist_ok=True)

    if not os.path.exists(SFT_OUTPUT_DIR):
        print("\n[1/5] Running SFT...")
        train_sft(
            model_id=BASE_MODEL_ID,
            dataset_path="gsm8k",
            formatting_function=format_gsm8k,
            output_dir=SFT_OUTPUT_DIR,
            log_file=os.path.join(LOGS_DIR, "sft_log.csv"),
            num_train_epochs=3,
            per_device_train_batch_size=BATCH_SIZE_TRAIN,
        )
    else:
        print("\n[1/5] SFT Model found, skipping training.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    if not os.path.exists(DPO_FULL_OUTPUT_DIR):
        print("\n[2/5] Running DPO (Full Dataset)...")
        train_dpo(
            model_id=SFT_OUTPUT_DIR,
            dataset_path="reciprocate/gsm8k_train_pairwise",
            dataset_split="train",
            formatting_function=format_dpo_data,
            output_dir=DPO_FULL_OUTPUT_DIR,
            log_file=os.path.join(LOGS_DIR, "dpo_full_log.csv"),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=8,
        )
    else:
        print("\n[2/5] DPO (Full) Model found, skipping training.")

    if not os.path.exists(DPO_1K_OUTPUT_DIR):
        print("\n[3/5] Running DPO (Truncated 1K)...")
        train_dpo(
            model_id=SFT_OUTPUT_DIR,
            dataset_path="reciprocate/gsm8k_train_pairwise",
            dataset_split="train[:1000]",
            formatting_function=format_dpo_data,
            output_dir=DPO_1K_OUTPUT_DIR,
            log_file=os.path.join(LOGS_DIR, "dpo_1k_log.csv"),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=8,
        )
    else:
        print("\n[3/5] DPO (1K) Model found, skipping training.")

    print("\n[4/5] Running Inference...")

    if not os.path.exists(RESULTS_BASE):
        print("   -> Generating Base Model...")
        generate_responses(
            model_path=BASE_MODEL_ID,
            dataset_path="gsm8k",
            output_file=RESULTS_BASE,
            batch_size=BATCH_SIZE_INFER,
        )

    if not os.path.exists(RESULTS_SFT):
        print("   -> Generating SFT Model...")
        generate_responses(
            model_path=SFT_OUTPUT_DIR,
            dataset_path="gsm8k",
            output_file=RESULTS_SFT,
            batch_size=BATCH_SIZE_INFER,
        )

    if not os.path.exists(RESULTS_DPO_FULL):
        print("   -> Generating DPO Model (Full)...")
        generate_responses(
            model_path=DPO_FULL_OUTPUT_DIR,
            dataset_path="gsm8k",
            output_file=RESULTS_DPO_FULL,
            batch_size=BATCH_SIZE_INFER,
        )

    if not os.path.exists(RESULTS_DPO_1K):
        print("   -> Generating DPO Model (1K)...")
        generate_responses(
            model_path=DPO_1K_OUTPUT_DIR,
            dataset_path="gsm8k",
            output_file=RESULTS_DPO_1K,
            batch_size=BATCH_SIZE_INFER,
        )

    print("\n[5/5] Running Evaluation (Math-Verify)...")

    print("\n--- Base Model Results ---")
    evaluate_file(RESULTS_BASE, EVAL_BASE)

    print("\n--- SFT Model Results ---")
    evaluate_file(RESULTS_SFT, EVAL_SFT)

    print("\n--- DPO Model Results (Full) ---")
    evaluate_file(RESULTS_DPO_FULL, EVAL_DPO_FULL)

    print("\n--- DPO Model Results (1K) ---")
    evaluate_file(RESULTS_DPO_1K, EVAL_DPO_1K)

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()