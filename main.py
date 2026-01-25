import os

from src.training.sft import train_sft
from src.training.dpo import train_dpo
from src.utils.evaluate import evaluate_file
from src.utils.inference import generate_responses
from src.utils.formatting_functions import format_gsm8k, format_dpo_data

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_OUTPUT_DIR = "./output/sft_model"
DPO_OUTPUT_DIR = "./output/dpo_model"
LOGS_DIR = "./output/logs"

RESULTS_BASE = "./output/results/base_results.jsonl"
RESULTS_SFT = "./output/results/sft_results.jsonl"
RESULTS_DPO = "./output/results/dpo_results.jsonl"

EVAL_BASE = "./output/results/base_summary.csv"
EVAL_SFT = "./output/results/sft_summary.csv"
EVAL_DPO = "./output/results/dpo_summary.csv"

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_INFER = 64
GRAD_ACCUM = 4
EPOCHS = 1


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_BASE), exist_ok=True)

    if not os.path.exists(SFT_OUTPUT_DIR):
        print("\n[1/4] Running SFT...")
        train_sft(
            model_id=BASE_MODEL_ID,
            dataset_path="gsm8k",
            formatting_function=format_gsm8k,
            output_dir=SFT_OUTPUT_DIR,
            log_file=os.path.join(LOGS_DIR, "sft_log.csv"),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE_TRAIN,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=2e-4,
        )
    else:
        print("\n[1/4] SFT Model found, skipping training.")

    if not os.path.exists(DPO_OUTPUT_DIR):
        print("\n[2/4] Running DPO (using SFT model as base)...")
        train_dpo(
            model_id=SFT_OUTPUT_DIR,
            dataset_path="reciprocate/gsm8k_train_pairwise",
            formatting_function=format_dpo_data,
            output_dir=DPO_OUTPUT_DIR,
            log_file=os.path.join(LOGS_DIR, "dpo_log.csv"),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=GRAD_ACCUM * 2,
            learning_rate=5e-6,
        )
    else:
        print("\n[2/4] DPO Model found, skipping training.")

    print("\n[3/4] Running Inference...")

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

    if not os.path.exists(RESULTS_DPO):
        print("   -> Generating DPO Model...")
        generate_responses(
            model_path=DPO_OUTPUT_DIR,
            dataset_path="gsm8k",
            output_file=RESULTS_DPO,
            batch_size=BATCH_SIZE_INFER,
        )

    print("\n[4/4] Running Evaluation (Math-Verify)...")

    print("\n--- Base Model Results ---")
    evaluate_file(RESULTS_BASE, EVAL_BASE)

    print("\n--- SFT Model Results ---")
    evaluate_file(RESULTS_SFT, EVAL_SFT)

    print("\n--- DPO Model Results ---")
    evaluate_file(RESULTS_DPO, EVAL_DPO)

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
