import os
from typing import Callable

import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

from peft import LoraConfig, TaskType
from trl import SFTTrainer


def train_sft(
    model_id: str,
    dataset_path: str,
    formatting_function: Callable,
    output_dir: str,
    log_file: str,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 32,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    dataset = load_dataset(dataset_path, "main", split="train")
    dataset = dataset.map(formatting_function, num_proc=16)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        bf16=True,
        optim="adamw_torch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        dataloader_num_workers=8,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(output_dir)

    if log_history := trainer.state.log_history:
        df_log = pd.DataFrame(log_history)
        df_log.to_csv(log_file, index=False)
