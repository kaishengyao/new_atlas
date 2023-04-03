# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from pathlib import Path
import argparse
import shutil
import tarfile
from datasets import load_dataset
def convert_anli(ex):
    return {"question": ex["premise"] + " entails " + ex["hypothesis"], "answers": ex["label"]}

def preprocess_anli(output_dir):

    data, index = {}, {}
    originaltrain = load_dataset('anli', split='train_r3')
    originaldev = load_dataset('anli', split='dev_r3')
    originaltest = load_dataset('anli', split='test_r3')

    data["train"] = [convert_anli(k) for k in originaltrain]
    data["dev"] = [convert_anli(k) for k in originaldev]
    data["test"] = [convert_anli(k) for k in originaltest]

    for split in data:
        with open(output_dir / (split + ".jsonl"), "w", encoding='utf-8') as fout:
            for ex in data[split]:
                try:
                    json.dump(ex, fout, ensure_ascii=False)
                    fout.write("\n")
                except:
                    continue

def main(args):
    output_dir = Path(args.output_directory)

    index_tar = output_dir / "index.tar"
    index_dir = output_dir / "dataindex"

    anli_dir = output_dir / "anli_data"

    if args.overwrite:
        print("Overwriting ANLI")
        download_anli = True
    else:
        download_anli = not anli_dir.exists()

    if download_anli:
        anli_dir.mkdir(parents=True, exist_ok=True)
        preprocess_anli(anli_dir)
    else:
        print("ANLI data already exists, not overwriting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data/",
        help="Path to the file to which the dataset is written.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite data")
    args = parser.parse_args()
    main(args)
