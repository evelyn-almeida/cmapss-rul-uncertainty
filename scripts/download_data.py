"""Download the NASA C-MAPSS turbofan dataset (FD001-FD004) into data/raw/.

Original source: NASA Prognostics Center of Excellence Data Set Repository
(Saxena et al., 2008). Files are fetched from a public GitHub mirror.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR

MIRROR = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        for prefix in ["train", "test", "RUL"]:
            name = f"{prefix}_{subset}.txt"
            target = DATA_DIR / name
            if not target.exists():
                urllib.request.urlretrieve(f"{MIRROR}/{name}", target)
                print(f"downloaded {name}")
    print(f"data ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
