import torch
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


def generate_responses(
    model_path: str,
    dataset_path: str,
    output_file: str,
    batch_size: int = 32,
    device: str = "auto",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Check if model_path is a LoRA adapter or base model
    is_adapter = False
    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        is_adapter = True
    except:
        is_adapter = False

    if is_adapter:
        # Load base model, then adapter
        base_model_id = peft_config.base_model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            device_map=device,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        # Load as regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )
    
    model.eval()

    dataset = load_dataset(dataset_path, "main", split="test")

    system_prompt = (
        "Solve the math problem step by step. "
        "At the end, strictly put your final answer within \\boxed{}, e.g., \\boxed{42}."
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Inference Batches"):
            batch = dataset[i : i + batch_size]
            questions = batch["question"]
            golds = batch["answer"]

            prompts = []
            for q in questions:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(text)

            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True
            ).to(model.device)

            with torch.inference_mode():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False, 
                        pad_token_id=tokenizer.pad_token_id,
                    )

            generated_ids = outputs[:, inputs.input_ids.shape[1] :]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for q, gold, resp in zip(questions, golds, responses):
                entry = {"question": q, "gold_answer": gold.split("####")[-1].strip(), "model_output": resp}
                f.write(json.dumps(entry) + "\n")
                f.flush()
