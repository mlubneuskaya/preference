import json
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from math_verify import parse, verify, LatexExtractionConfig


def verify_single_sample(item):

    model_out = item["model_output"]
    raw_gold = item["gold_answer"]

    try:
        parsed_pred = parse(model_out)
        parsed_gold = parse(raw_gold)
        is_correct = verify(parsed_gold, parsed_pred)
    except Exception as e:
        print(e)
        parsed_pred = []
        is_correct = False

    model_extracted = parsed_pred[0] if isinstance(parsed_pred, list) and len(parsed_pred) > 0 else ""

    return {
        "question": item.get("question"),
        "correct": is_correct,
        "gold_extracted": parsed_gold,
        "model_extracted": model_extracted,
    }


def evaluate_file(input_file, output_file, num_workers=None):
    data = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if num_workers is None:
        try:
            num_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            num_workers = os.cpu_count() or 1

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(verify_single_sample, data), total=len(data)))

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
