import json
import os
import argparse
from evaluate import load

def calculate_bleu(results_dir, output_dir=None, lang_tag="en"):
    # Load data
    with open(results_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    predictions = []
    references = []

    for line in lines:
        data = json.loads(line)
        predictions.append(data["prediction"])
        references.append([data["answer"]])  # Wrap in a list as BLEU expects list of references per prediction

    # Load BLEU metric
    bleu = load("bleu")
    result = bleu.compute(predictions=predictions, references=references)

    # Display and optionally save
    print(f"[BLEU-{lang_tag}] Score: {result['bleu']:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"bleu_score_{lang_tag}.json")
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(result, out_file, indent=2)
        print(f"Saved BLEU score to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the JSON results file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save BLEU result")
    parser.add_argument("--lang_tag", type=str, required=True, help="Language tag for output naming")
    args = parser.parse_args()

    calculate_bleu(args.results_dir, args.output_dir, args.lang_tag)