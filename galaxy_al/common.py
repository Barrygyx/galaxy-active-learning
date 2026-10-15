"""Small I/O helpers. Writes use atomic replacement; inputs stay unchanged."""
import csv
import hashlib
import json
import os
from pathlib import Path


CLASS_NAMES = [
    "Disturbed", "Merging", "Round smooth", "In-between round smooth",
    "Cigar shaped smooth", "Barred spiral", "Unbarred tight spiral",
    "Unbarred loose spiral", "Edge-on without bulge", "Edge-on with bulge",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns=None):
    rows = list(rows)
    if columns is None:
        columns = list(rows[0]) if rows else []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def code_hashes(names):
    folder = Path(__file__).resolve().parent
    return {name: sha256_file(folder / name) for name in names}
