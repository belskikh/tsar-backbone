import argparse

from functools import partial
from multiprocessing import Pool
from pathlib import Path

from typing import Union, List

from tqdm import tqdm


import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--dir", help="train or val_imgs", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--workers", default=10)
    return parser.parse_args()


def get_labels(target_df: pd.DataFrame,
               aggregate_df: pd.DataFrame) -> pd.DataFrame:

    def apply_fn(row: pd.Series) -> str:
        try:
            labels = aggregate_df[row["fpath"].stem]
        except:
            labels = ""
        return labels

    return target_df.progress_apply(apply_fn, axis=1)


def main(args: argparse.Namespace) -> None:
    tqdm.pandas()

    print("Load dataframe...")
    in_df = pd.read_csv(args.in_csv)

    root = Path(args.dir)

    pbar = tqdm()
    images = []
    for d in root.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix == ".jpg":
                    images.append(f)
                    pbar.update()
    pbar.close()

    result_df = pd.DataFrame(data={"fpath": images})

    aggregate_df = in_df.groupby("ImageID")["LabelName"].aggregate(
        lambda x: " ".join(list(x))
    )

    workers = args.workers
    map_function = partial(get_labels, aggregate_df=aggregate_df)
    with Pool(workers) as p:
        res = p.map(map_function, np.array_split(result_df, workers))

    result_df["labels"] = pd.concat(res)
    result_df.to_csv(args.out_csv, index = None)


if __name__ == "__main__":
    args = parse_args()
    main(args)
