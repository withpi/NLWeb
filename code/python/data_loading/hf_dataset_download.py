#!/usr/bin/env python3
"""
Utility to download a dataset from Hugging Face and save it in tab-separated format.
"""

import argparse
import json
from datasets import load_dataset


def download_dataset(split_name, output_path=None):
    """
    Download a split from withpi/nlweb dataset and save as JSONL.
    Format: url\tschema_object per line.

    Args:
        split_name: Name of the split to download (e.g., 'allbirds')
        output_path: Path where the output file should be saved (default: {split_name}.jsonl)
    """
    print(f"Loading dataset split '{split_name}' from Hugging Face...")
    dataset = load_dataset("withpi/nlweb", "all", split=split_name)

    print(f"Loaded {len(dataset)} records")

    output_file = output_path if output_path else f"{split_name}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for record in dataset:
            url = record["url"]
            schema_object = json.loads(record["schema_object"])

            # If schema_object is a dict, convert to JSON string
            if isinstance(schema_object, dict):
                schema_object = json.dumps(schema_object, ensure_ascii=False, separators=(',', ':'))

            f.write(f"{url}\t{schema_object}\n")

    print(f"Saved {len(dataset)} records to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a dataset split from Hugging Face withpi/nlweb"
    )
    parser.add_argument(
        "split_name",
        help="Name of the split to download (e.g., 'allbirds')"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: {split_name}.jsonl)"
    )
    args = parser.parse_args()

    download_dataset(args.split_name, args.output)
