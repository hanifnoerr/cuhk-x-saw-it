"""CSV contracts and scoring shared by inference and training."""
import csv, hashlib, json
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"single", "multi", "combination", "sequence", "emotion", "object_interaction"}

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Duplicate columns: {path}")
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f"Malformed row: {path}")
        return rows


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_path(row):
    path = PurePosixPath(row["path"])
    return str(path.parent.parent if path.suffix.lower() == ".mp4" else path)


def validate_answer(row, answer):
    category = row["category"]
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    available = {x for x in "ABCD" if row.get(x, "").strip()}
    if not answer or set(answer) - available or len(answer) != len(set(answer)):
        raise ValueError(f"Invalid answer {answer!r} for {row['qa_id']}")
    if category == "sequence":
        if set(answer) != available:
            raise ValueError(f"Incomplete sequence for {row['qa_id']}")
    elif category != "multi" and len(answer) != 1:
        raise ValueError(f"Expected one letter for {row['qa_id']}")


def correct(row, prediction):
    validate_answer(row, prediction)
    validate_answer(row, row["answer"])
    if row["category"] == "multi":
        return set(prediction) == set(row["answer"])
    return prediction == row["answer"]


def validate_submission(rows, questions, sample):
    fields = list(sample[0])
    if len(fields) != 2 or fields[0] != "qa_id":
        raise ValueError("Unexpected sample submission schema")
    if any(list(r) != fields for r in rows):
        raise ValueError(f"Expected columns {fields}")
    ids = [r["qa_id"] for r in rows]
    expected = [r["qa_id"] for r in sample]
    if len(ids) != len(set(ids)) or ids != expected:
        raise ValueError("Submission IDs must exactly match sample IDs and order")
    lookup = {r["qa_id"]: r for r in questions}
    if len(lookup) != len(questions) or set(ids) != set(lookup):
        raise ValueError("Question/sample ID mismatch")
    for row in rows:
        validate_answer(lookup[row["qa_id"]], row[fields[1]])
    return {"rows": len(rows), "columns": fields, "status": "valid_local_contract"}
