from email.mime import text
import re


def format_gsm8k(example):
    prompt = (
        "Solve the math problem step by step. "
        "At the end, strictly put your final answer within \\boxed{}, e.g., \\boxed{42}."
    )

    raw_answer = example["answer"]
    parts = raw_answer.split("####")
    reasoning = parts[0].strip()
    
    reasoning = re.sub(r"<<.*?>>", "", reasoning)
    final_value = parts[1].strip()
    
    formatted_answer = f"<think>\n{reasoning}\n</think>\n\nThe final answer is \\boxed{{{final_value}}}."
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": formatted_answer},
    ]
    return {"messages": messages}


def format_dpo_data(example):

    def format(text):
        if "####" in text:
            reasoning, final_answer = text.split("####")
            return [{"role": "assistant", "content": f"<think>\n{reasoning.strip()}\n</think>\n\nThe final answer is \\boxed{{{final_answer.strip()}}}."}]
        return [{"role": "assistant", "content": text}]

    question = example.get("prompt")
    chosen = example.get("selected")
    rejected = example.get("rejected")

    system_prompt = (
        "Solve the math problem step by step. "
        "At the end, strictly put your final answer within \\boxed{}, e.g., \\boxed{42}."
    )
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    return {"prompt": prompt, "chosen": format(chosen), "rejected": format(rejected)}
