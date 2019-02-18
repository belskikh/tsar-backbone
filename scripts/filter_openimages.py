import argparse

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

from typing import List

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--trainable-classes", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--workers", default=10)

    return parser.parse_args()


def filter_trainable(df: pd.DataFrame,
                     trainable_classes: List[str]) -> pd.DataFrame:

    def apply_fn(row: pd.Series) -> bool:
        return (row["LabelName"] in trainable_classes and row[
            "Confidence"] == 1)

    return df[df.progress_apply(apply_fn, axis=1)]


def main(args: argparse.Namespace):
    tqdm.pandas()
    trainable_classes = [s for s in pd.read_csv(args.trainable_classes,
                                                header=None)[0]]

    print("Load dataframe...")
    in_df = pd.read_csv(args.in_csv)
    print("Dataframe loaded. Start filtering...")

    workers = args.workers
    filter_fn = partial(filter_trainable, trainable_classes=trainable_classes)
    with Pool(workers) as p:
        res = list(p.imap(filter_fn, np.array_split(in_df, workers)))

    print("Filtering done. Concat results...")
    trainable_df = pd.concat(res)

    print("Concating done. Save...")
    trainable_df.to_csv(args.out_csv, index=None)


if __name__ == "__main__":
    args = parse_args()
    main(args)
