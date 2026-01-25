def format_gsm8k(example):
    prompt = (
        "You are a helpful math assistant. "
        "Solve the problem step-by-step and output the final answer within \\boxed{}."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return {"messages": messages}


def format_dpo_data(example, tokenizer):
    question = example.get("question") or example.get("prompt")
    chosen = example.get("chosen") or example.get("response_j")
    rejected = example.get("rejected") or example.get("response_k")

    if not question or not chosen or not rejected:
        raise ValueError(
            f"Could not find expected columns in example: {example.keys()}"
        )

    system_prompt = (
        "You are a helpful math assistant. "
        "Solve the problem step-by-step and output the final answer within \\boxed{}."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return {"prompt": formatted_prompt, "chosen": chosen, "rejected": rejected}
