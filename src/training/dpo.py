import os
from typing import Callable
from functools import partial

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType
from trl import DPOTrainer


def train_dpo(
    model_id: str,
    dataset_path: str,
    formatting_function: Callable,
    output_dir: str,
    log_file: str,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 32,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    beta: float = 0.1,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    dataset = load_dataset(dataset_path, split="train")

    formatted_dataset = dataset.map(
        partial(formatting_function, tokenizer=tokenizer),
        remove_columns=dataset.column_names,
        num_proc=16,
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        dataloader_num_workers=8,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=beta,
        max_length=1024,
        max_prompt_length=512,
    )

    trainer.train()

    trainer.save_model(output_dir)

    if trainer.state.log_history:
        df_log = pd.DataFrame(trainer.state.log_history)
        df_log.to_csv(log_file, index=False)
