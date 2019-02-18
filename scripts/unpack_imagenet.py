import argparse
import tarfile

from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    parser.add_argument("--workers", type=int, default=0)

    return parser.parse_args()


def untar(f, dir):
    t = tarfile.open(f)
    extract_dir = dir / f.stem
    extract_dir.mkdir(exist_ok=True)

    t.extractall(extract_dir)
    t.close()
    f.unlink()


def main(args):
    dir = Path(args.dir)
    files = [f for f in dir.iterdir() if f.suffix == ".tar"]
    workers = args.workers

    untar_fn = partial(untar, dir=dir)

    if workers:
        with Pool(workers) as p:
            total = len(files)
            with tqdm(total=total) as pbar:
                for i, _ in enumerate(p.imap_unordered(untar_fn, files)):
                    pbar.update()
    else:
        for f in tqdm(files):
            untar_fn(f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
