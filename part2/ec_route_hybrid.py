import argparse
from contextlib import closing
from io import BytesIO
from pathlib import Path
import pickle
import re
from statistics import mean
from zipfile import ZipFile

from train_ec_entity_mask import EntitySlots, MODEL_ID, generate_sql, open_database, read_lines


BASELINE_TEST = Path("submission_ec_scratch_epoch60")
MASKED_BEST = Path("checkpoints/scr_experiments/ec_entity_mask/best")
OUTPUT = Path("submission_ec_route_hybrid")
TEST_STEM = "t5_ft_experiment_ec_test"

ROUTE = re.compile(r"\bfrom\s+city <extra_id_\d+>\s+to\s+city <extra_id_\d+>[\s?.!]*$", re.I)
FLIGHT = re.compile(r"\bflights?\b", re.I)
NUMBER = re.compile(r"(?<!_)\b\d")
CONSTRAINT = re.compile(
    r"\b(?:before|after|at|by|daily|round|one\s+way|nonstop|direct|stop|stops|"
    r"departure|depart|leaving|arrive|arriving|arrives|arrival|layover|connecting|"
    r"connection|through|via|return|lowest|cheapest|earliest|latest|last|first|"
    r"early|late|least|most|only|airline|airport|aircraft|fare|fares|meal|"
    r"breakfast|lunch|dinner|cost|price|time|times|hours|classes|class|coach|"
    r"business|seats|dollars|today|tomorrow|morning|evening|afternoon|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|"
    r"march|april|may|june|july|august|september|october|november|december)\b",
    re.I,
)


class RecordsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Unexpected class in records: {module}.{name}")


def read_records(data):
    return RecordsUnpickler(BytesIO(data)).load()


def simple_route(slots, question):
    text, values = slots.encode_question(question)
    return (
        len(values) == 2
        and ROUTE.search(text) is not None
        and FLIGHT.search(text) is not None
        and CONSTRAINT.search(text) is None
        and NUMBER.search(text) is None
    )


def record_f1(gold, predicted):
    gold, predicted = set(gold), set(predicted)
    overlap = len(gold & predicted)
    precision = overlap / len(predicted) if predicted else 1
    recall = overlap / len(gold) if gold else 1
    return 2 * precision * recall / (precision + recall + 1e-8)


def evaluate_development(slots, archive_path):
    questions = read_lines(Path("data/dev.nl"))
    gold, _ = read_records(Path("records/ground_truth_dev.pkl").read_bytes())
    with ZipFile(archive_path) as archive:
        original, original_errors = read_records(archive.read("dev_epoch_60.pkl"))
        masked, masked_errors = read_records(archive.read("dev_epoch_50.pkl"))
    if len({len(x) for x in (
        questions, gold, original, original_errors, masked, masked_errors,
    )}) != 1:
        raise ValueError("Development result lengths do not match.")
    routes = [simple_route(slots, question) for question in questions]
    chosen = [route and not error for route, error in zip(routes, masked_errors)]
    hybrid = [b if use_mask else a for a, b, use_mask in zip(original, masked, chosen)]
    for label, predictions in (("Original", original), ("Entity-slot", masked), ("Hybrid", hybrid)):
        score = mean(record_f1(g, p) for g, p in zip(gold, predictions))
        print(f"{label} dev Record F1 {score:.2%}")
    print(f"Simple routes: {sum(routes)}/{len(routes)} | switched: {sum(chosen)}")


def predict_test(slots, database):
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    from utils import compute_records

    questions = read_lines(Path("data/test.nl"))
    sql = read_lines(BASELINE_TEST / f"{TEST_STEM}.sql")
    records, errors = read_records((BASELINE_TEST / f"{TEST_STEM}.pkl").read_bytes())
    if len({len(questions), len(sql), len(records), len(errors)}) != 1:
        raise ValueError("Baseline predictions must align with the test questions.")
    indices = [i for i, question in enumerate(questions) if simple_route(slots, question)]
    switched = 0
    if indices:
        if not torch.cuda.is_available():
            raise SystemExit("Run test prediction on a GPU compute node.")
        model = T5ForConditionalGeneration.from_pretrained(MASKED_BEST).to("cuda")
        tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID)
        candidates = list(generate_sql(
            model, tokenizer, slots, [questions[i] for i in indices], database, 8, torch
        ))
        candidate_records, candidate_errors = compute_records(candidates)
        for index, query, rows, error in zip(
            indices, candidates, candidate_records, candidate_errors, strict=True
        ):
            if not error:
                sql[index], records[index], errors[index] = query, rows, error
                switched += 1
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / f"{TEST_STEM}.sql").write_text("\n".join(sql) + "\n", encoding="utf-8")
    with (OUTPUT / f"{TEST_STEM}.pkl").open("wb") as stream:
        pickle.dump((records, errors), stream)
    print(f"Test predictions {len(sql)} | simple routes {len(indices)} | switched {switched}")
    print(f"Saved to {OUTPUT}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check-dev", "predict-test"))
    parser.add_argument("--archive", type=Path, default=Path("ec_dev_compare.zip"))
    args = parser.parse_args()
    with closing(open_database(Path.cwd())) as database:
        slots = EntitySlots(database)
        if args.mode == "check-dev":
            evaluate_development(slots, args.archive)
        else:
            predict_test(slots, database)


if __name__ == "__main__":
    main()
