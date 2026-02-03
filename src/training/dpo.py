# src/training/dpo.py

import os
from typing import Callable
from functools import partial
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, PeftModel, PeftConfig
from trl import DPOTrainer, DPOConfig

def train_dpo(
        model_id: str,
        dataset_path: str,
        formatting_function: Callable,
        output_dir: str,
        log_file: str,
        dataset_split: str = "train", 
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-6,
        beta: float = 0.1,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
):   
    adapter_config = PeftConfig.from_pretrained(model_id)
    base_model_path = adapter_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(model, model_id, is_trainable=True)
    model = model.merge_and_unload()

    dataset = load_dataset(dataset_path, split=dataset_split)

    formatted_dataset = dataset.map(
        formatting_function,
        remove_columns=dataset.column_names,
        num_proc=16
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

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        learning_rate=learning_rate,
        beta=beta,
        max_length=1024,
        max_prompt_length=512,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(output_dir)

    if trainer.state.log_history:
        df_log = pd.DataFrame(trainer.state.log_history)
        df_log.to_csv(log_file, index=False)
    
    tokenizer.save_pretrained(output_dir)
