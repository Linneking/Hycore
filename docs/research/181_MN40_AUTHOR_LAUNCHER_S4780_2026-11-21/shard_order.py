"""Pin the two author dataset glob patterns to their preregistered shard order."""

import glob
import json
import os
from pathlib import Path
import sys


def install(order_file):
    orders = json.loads(Path(order_file).read_text())
    natural_glob = glob.glob

    def ordered_glob(pathname, *args, **kwargs):
        files = natural_glob(pathname, *args, **kwargs)
        order = orders.get(os.fspath(pathname))
        if order is None:
            return files
        names = [os.path.basename(path) for path in files]
        if names == order:
            return files
        if len(names) != len(order) or set(names) != set(order):
            raise RuntimeError(f"Unexpected shards in {pathname}: {names}")
        paths_by_name = dict(zip(names, files))
        return [paths_by_name[name] for name in order]

    glob.glob = ordered_glob


def verify(order_file):
    orders = json.loads(Path(order_file).read_text())
    for pattern, expected in orders.items():
        result = [os.path.basename(path) for path in glob.glob(pattern)]
        if result != expected:
            raise RuntimeError(f"Shard order hook did not activate for {pattern}: {result}")
    print(json.dumps(orders))


if __name__ == "__main__":
    verify(sys.argv[1])
