import os
from typing import Callable

import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


def train_sft(
    model_id: str,
    dataset_path: str,
    formatting_function: Callable,
    output_dir: str,
    log_file: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    dataset = load_dataset(dataset_path, "main", split="train")
    dataset = dataset.map(formatting_function)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
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

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_dir=os.path.join(output_dir, "logs"),
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=False,
        bf16=True,
        dataset_text_field="messages",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(output_dir)

    if log_history := trainer.state.log_history:
        df_log = pd.DataFrame(log_history)
        df_log.to_csv(log_file, index=False)
