import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import pickle
import re
import sqlite3
from statistics import mean
import time
from types import SimpleNamespace

from ec_route_hybrid import read_records, record_f1
from train_ec_entity_mask import MODEL_ID, SQL_LITERAL, open_database, read_lines


NAME = "t5_ft_experiment_test"
SUBMISSION = Path("submission_continue_lr5e_5")
OUTPUT = Path("submission_q7_time_repair")
CHECKPOINT = Path("checkpoints/ft_experiments/continue_lr5e_5/best")
DEV_OUTPUT = Path("results/q7_time_repair_dev")

# The common ATIS departure windows in the training annotations.
PERIOD_RANGES = {
    "morning": (0, 1200), "early morning": (0, 800),
    "afternoon": (1200, 1800), "early afternoon": (1200, 1400),
    "late afternoon": (1600, 1800), "evening": (1800, 2200),
}
PERIOD = re.compile(r"\b(?:(?:early|late)\s+)?(?:morning|afternoon|evening)\b", re.I)
CLOCK = r"(?P<hour>\d{1,4})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)"
EXPLICIT = re.compile(r"\b(?P<op>before|after|around|about|near|approximately)\s+" + CLOCK + r"\b", re.I)
ANY_CLOCK = re.compile(r"\b\d{1,4}(?::\d{2})?\s*(?:am|pm)\b", re.I)
TIME_CONDITION = re.compile(
    r"\b(?P<column>flight_\d+\.(?:departure|arrival)_time)\s*"
    r"(?:BETWEEN\s+\d+\s+AND\s+\d+|(?:>=|<=|=|>|<)\s*\d+"
    r"(?:\s+AND\s+(?P=column)\s*(?:>=|<=|=|>|<)\s*\d+)?)(?![\d.])",
    re.I,
)
ARRIVAL = re.compile(r"\b(?:arriv\w*|land\w*)\b", re.I)
DEPARTURE = re.compile(r"\b(?:depart\w*|leav\w*|take\s+off)\b", re.I)
SELECT = re.compile(r"\bselect\b", re.I)
COMPOUND = re.compile(r"\b(?:or|union|intersect|except)\b", re.I)


def parse_clock(match):
    hour = int(match["hour"])
    minute = int(match["minute"] or 0)
    if len(match["hour"]) > 2 and match["minute"] is None:
        hour, minute = divmod(hour, 100)
    if 1 <= hour <= 12 and 0 <= minute < 60:
        return (hour % 12 + (12 if match["meridiem"].lower() == "pm" else 0)) * 60 + minute
    return None


def hhmm(minutes):
    hour, minute = divmod(minutes, 60)
    return hour * 100 + minute


def repair_time_condition(question, sql):
    unquoted = SQL_LITERAL.sub("''", sql)
    if len(SELECT.findall(unquoted)) != 1 or COMPOUND.search(unquoted):
        return sql
    conditions = list(TIME_CONDITION.finditer(sql))
    if len(conditions) != 1:
        return sql

    match = conditions[0]
    column = match["column"]
    arrival = bool(ARRIVAL.search(question))
    departure = bool(DEPARTURE.search(question))
    if (arrival and departure or arrival and "arrival_time" not in column.lower()
            or departure and "departure_time" not in column.lower()):
        return sql

    periods = PERIOD.findall(question)
    explicit = list(EXPLICIT.finditer(question))
    clocks = ANY_CLOCK.findall(question)
    if len(periods) == 1 and not clocks and not arrival and column.lower().endswith(".departure_time"):
        bounds = PERIOD_RANGES.get(" ".join(periods[0].lower().split()))
        if bounds is None:
            return sql
        replacement = f"{column} BETWEEN {bounds[0]} AND {bounds[1]}"
    elif len(explicit) == len(clocks) == 1 and not periods:
        clock = parse_clock(explicit[0])
        if clock is None:
            return sql
        operator = explicit[0]["op"].lower()
        if operator in ("before", "after"):
            replacement = f"{column} {'<' if operator == 'before' else '>'} {hhmm(clock)}"
        elif 30 <= clock < 24 * 60 - 30:
            replacement = f"{column} BETWEEN {hhmm(clock - 30)} AND {hhmm(clock + 30)}"
        else:
            return sql
    else:
        return sql
    normalize = lambda value: re.sub(r"\s+", "", value.casefold())
    if normalize(match.group()) == normalize(replacement):
        return sql
    return sql[:match.start()] + replacement + sql[match.end():]


def apply_repairs(root, questions, sql, records, errors):
    if len({len(questions), len(sql), len(records), len(errors)}) != 1:
        raise ValueError("Questions, SQL, and records must have the same length.")
    edits = []
    for index, (question, original) in enumerate(zip(questions, sql, strict=True)):
        repaired = repair_time_condition(question, original)
        if repaired != original:
            edits.append((index, repaired))

    def execute(edit):
        index, repaired = edit
        with closing(open_database(root)) as database:
            deadline = time.monotonic() + 5
            database.set_progress_handler(lambda: time.monotonic() >= deadline, 1_000)
            try:
                return index, repaired, database.execute(repaired).fetchall()
            except sqlite3.Error:
                return index, repaired, None

    applied = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, repaired, rows in pool.map(execute, edits):
            if rows is not None:
                sql[index] = repaired
                records[index] = rows
                errors[index] = ""
                applied += 1
    print(f"Time condition changes proposed: {len(edits)} | applied: {applied}")


def save(output, name, sql, records, errors):
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.sql").write_text("\n".join(sql) + "\n", encoding="utf-8")
    with (output / f"{name}.pkl").open("wb") as stream:
        pickle.dump((records, errors), stream)
    print(f"Saved {len(sql)} SQL predictions and records to {output / name}.*")


def check_dev(root, sql_path, records_path):
    questions = read_lines(root / "data/dev.nl")
    gold, _ = read_records((root / "records/ground_truth_dev.pkl").read_bytes())
    sql = read_lines(sql_path)
    rows, errors = read_records(records_path.read_bytes())
    before = [record_f1(g, r) for g, r in zip(gold, rows, strict=True)]
    apply_repairs(root, questions, sql, rows, errors)
    after = [record_f1(g, r) for g, r in zip(gold, rows, strict=True)]
    print(f"Original dev Record F1: {mean(before):.2%}")
    print(f"Repaired dev Record F1: {mean(after):.2%}")
    print(f"Improved: {sum(b > a + 1e-8 for a, b in zip(before, after))} | "
          f"Worsened: {sum(b < a - 1e-8 for a, b in zip(before, after))}")
    return sql, rows, errors


def validate_final_dev(root):
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    from load_data import get_dataloader
    from train_t5 import eval_epoch

    if not torch.cuda.is_available():
        raise SystemExit("Use an HPC GPU terminal to regenerate the final model's dev predictions.")
    model = T5ForConditionalGeneration.from_pretrained(root / CHECKPOINT).to("cuda").eval()
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID)
    original_generate = model.generate
    beams = 8
    directory = root / DEV_OUTPUT
    directory.mkdir(parents=True, exist_ok=True)
    sql_path = directory / "original.sql"
    records_path = directory / "original.pkl"

    def generate_with_sql_check(*inputs, **kwargs):
        kwargs.update(num_beams=beams, num_return_sequences=beams, do_sample=False)
        candidates = original_generate(*inputs, **kwargs)
        queries = tokenizer.batch_decode(candidates, skip_special_tokens=True)
        selected = []
        for start in range(0, len(queries), beams):
            best = start
            for index in range(start, start + beams):
                try:
                    database.execute("EXPLAIN " + queries[index].strip()).close()
                except sqlite3.Error:
                    continue
                best = index
                break
            selected.append(candidates[best])
        return torch.stack(selected)

    with closing(open_database(root)) as database:
        model.generate = generate_with_sql_check
        eval_epoch(
            SimpleNamespace(finetune=True, experiment_name="continue_lr5e_5"),
            model, get_dataloader(4, "dev"), "data/dev.sql", str(sql_path),
            "records/ground_truth_dev.pkl", str(records_path),
        )
    save(directory, "repaired", *check_dev(root, sql_path, records_path))


def predict_test(root):
    questions = read_lines(root / "data/test.nl")
    stem = root / SUBMISSION / NAME
    sql = read_lines(stem.with_suffix(".sql"))
    records, errors = read_records(stem.with_suffix(".pkl").read_bytes())
    apply_repairs(root, questions, sql, records, errors)
    save(root / OUTPUT, NAME, sql, records, errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate-dev", "check-dev", "predict-test"))
    parser.add_argument("--sql", type=Path)
    parser.add_argument("--records", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "data/flight_database.db").is_file():
        parser.error("Run from the assignment's part2 directory.")
    if args.mode == "validate-dev":
        validate_final_dev(root)
    elif args.mode == "check-dev":
        if args.sql is None or args.records is None:
            parser.error("check-dev requires --sql and --records.")
        check_dev(root, args.sql, args.records)
    else:
        predict_test(root)


if __name__ == "__main__":
    main()
