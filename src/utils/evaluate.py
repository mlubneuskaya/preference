import json
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from math_verify import parse, verify, LatexExtractionConfig


def verify_single_sample(item):
    config = LatexExtractionConfig(
        normalization_config={"lowercase": True}, boxed_match_priority=0
    )

    model_out = item.get("model_output", "")
    raw_gold = item.get("gold_answer", "")

    if raw_gold and "####" in raw_gold:
        clean_gold = raw_gold.split("####")[1].strip()
    else:
        clean_gold = raw_gold

    try:
        parsed_pred = parse(model_out, extraction_config=config)
        parsed_gold = parse(clean_gold, extraction_config=config)
        is_correct = verify(parsed_gold, parsed_pred)
    except Exception as e:
        parsed_pred = str(e)
        is_correct = False

    return {
        "question": item.get("question"),
        "correct": is_correct,
        "gold_extracted": clean_gold,
        "model_extracted": str(parsed_pred),
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
