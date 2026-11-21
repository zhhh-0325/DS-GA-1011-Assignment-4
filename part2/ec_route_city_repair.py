import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import pickle
import re
from statistics import mean
import sqlite3
import time
from zipfile import ZipFile

from ec_route_hybrid import OUTPUT as HYBRID_OUTPUT
from ec_route_hybrid import TEST_STEM, read_records, record_f1, simple_route
from train_ec_entity_mask import EntitySlots, SQL_LITERAL, open_database, read_lines


OUTPUT = Path("submission_ec_route_grounded")
IDENTIFIER = r"[a-z_]\w*"
CITY_SLOT = r"city\s+(<extra_id_\d+>)"
CITY_MENTION = re.compile(CITY_SLOT, re.I)
QUESTION_ROUTE = re.compile(r"\bfrom\s+" + CITY_SLOT + r"\s+to\s+" + CITY_SLOT, re.I)
AMBIGUOUS_ROUTE = re.compile(r"\b(?:either|or|via|through|round|return|back|between)\b", re.I)
SELECT = re.compile(r"\bselect\b", re.I)
COMPOUND_SQL = re.compile(r"\b(?:or|union|intersect|except)\b", re.I)


def repair_city_conditions(question, sql, slots):
    text, bindings = slots.encode_question(question)
    routes = QUESTION_ROUTE.findall(text)
    if (
        len(routes) != 1
        or len(set(CITY_MENTION.findall(text))) != 2
        or AMBIGUOUS_ROUTE.search(text)
    ):
        return sql

    unquoted = SQL_LITERAL.sub("''", sql)
    if len(SELECT.findall(unquoted)) != 1 or COMPOUND_SQL.search(unquoted):
        return sql

    edits = []
    flight_aliases = set()
    for direction, slot in zip(("from", "to"), routes[0]):
        join = re.findall(
            rf"\b({IDENTIFIER})\.{direction}_airport\s*=\s*({IDENTIFIER})\.airport_code\b",
            unquoted, re.I,
        )
        if len(join) != 1:
            return sql
        flight_alias, service_alias = join[0]
        flight_aliases.add(flight_alias.casefold())
        city_alias = re.findall(
            rf"\b{re.escape(service_alias)}\.city_code\s*=\s*({IDENTIFIER})\.city_code\b",
            unquoted, re.I,
        )
        if len(city_alias) != 1:
            return sql
        field = rf"\b{re.escape(city_alias[0])}\.city_name\s*=\s*"
        values = list(re.finditer(field + "(" + SQL_LITERAL.pattern + ")", sql, re.I))
        if len(values) != 1:
            return sql
        start, end = values[0].span(1)
        value = "'" + bindings[slot].replace("'", "''") + "'"
        edits.append((start, end, value))

    if len(flight_aliases) != 1 or edits[0][0] == edits[1][0]:
        return sql
    for start, end, value in sorted(edits, reverse=True):
        sql = sql[:start] + value + sql[end:]
    return sql


def apply_repairs(root, questions, sql, records, errors, slots):
    if len({len(questions), len(sql), len(records), len(errors)}) != 1:
        raise ValueError("Questions, SQL, and records must have the same length.")
    edits = []
    for index, (question, original) in enumerate(zip(questions, sql, strict=True)):
        repaired = repair_city_conditions(question, original, slots)
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

    accepted = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, repaired, rows in pool.map(execute, edits):
            if rows is not None:
                sql[index] = repaired
                records[index] = rows
                errors[index] = ""
                accepted += 1
    print(f"City conditions proposed: {len(edits)} | applied: {accepted}")


def check_dev(root, archive_path, slots):
    questions = read_lines(root / "data/dev.nl")
    gold, _ = read_records((root / "records/ground_truth_dev.pkl").read_bytes())
    with ZipFile(archive_path) as archive:
        base_sql = archive.read("dev_epoch_60.sql").decode("utf-8").splitlines()
        mask_sql = archive.read("dev_epoch_50.sql").decode("utf-8").splitlines()
        base_rows, base_errors = read_records(archive.read("dev_epoch_60.pkl"))
        mask_rows, mask_errors = read_records(archive.read("dev_epoch_50.pkl"))
    use_mask = [simple_route(slots, question) and not error
                for question, error in zip(questions, mask_errors, strict=True)]
    sql, rows, errors = [
        [b if switch else a for a, b, switch in zip(old, new, use_mask, strict=True)]
        for old, new in ((base_sql, mask_sql), (base_rows, mask_rows), (base_errors, mask_errors))
    ]
    before = [record_f1(g, r) for g, r in zip(gold, rows, strict=True)]
    apply_repairs(root, questions, sql, rows, errors, slots)
    after = [record_f1(g, r) for g, r in zip(gold, rows, strict=True)]
    print(f"Hybrid dev Record F1 {mean(before):.2%}")
    print(f"City-grounded dev Record F1 {mean(after):.2%}")
    print(
        f"Improved: {sum(b > a + 1e-8 for a, b in zip(before, after))} | "
        f"Worsened: {sum(b < a - 1e-8 for a, b in zip(before, after))}"
    )


def predict_test(root, slots):
    questions = read_lines(root / "data/test.nl")
    stem = root / HYBRID_OUTPUT / TEST_STEM
    sql = read_lines(stem.with_suffix(".sql"))
    records, errors = read_records(stem.with_suffix(".pkl").read_bytes())
    apply_repairs(root, questions, sql, records, errors, slots)
    destination = root / OUTPUT
    destination.mkdir(exist_ok=True)
    (destination / f"{TEST_STEM}.sql").write_text("\n".join(sql) + "\n", encoding="utf-8")
    with (destination / f"{TEST_STEM}.pkl").open("wb") as stream:
        pickle.dump((records, errors), stream)
    print(f"Saved {len(sql)} predictions and records to {OUTPUT}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check-dev", "predict-test"))
    parser.add_argument("--archive", type=Path, default=Path("ec_dev_compare.zip"))
    args = parser.parse_args()
    root = Path.cwd()
    with closing(open_database(root)) as database:
        slots = EntitySlots(database)
    if args.mode == "check-dev":
        check_dev(root, args.archive, slots)
    else:
        predict_test(root, slots)


if __name__ == "__main__":
    main()
