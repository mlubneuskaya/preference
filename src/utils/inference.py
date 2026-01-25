import torch
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    dataset = load_dataset(dataset_path, "main", split="test")

    system_prompt = (
        "You are a helpful math assistant. "
        "Solve the problem step-by-step and output the final answer within \\boxed{}."
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
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_ids = outputs[:, inputs.input_ids.shape[1] :]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for q, gold, resp in zip(questions, golds, responses):
                entry = {"question": q, "gold_answer": gold, "model_output": resp}
                f.write(json.dumps(entry) + "\n")
                f.flush()
