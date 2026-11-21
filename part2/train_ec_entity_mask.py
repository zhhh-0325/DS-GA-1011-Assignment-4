import argparse
from collections import defaultdict
from contextlib import closing
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace


MODEL_ID = "google-t5/t5-small"
SQL_LITERAL = re.compile(r"'(?:[^']|'')*'")
SLOT = re.compile(r"<extra_id_\d+>")
AIRLINE_SUFFIX = re.compile(r"(?:\s+(?:airlines?|airways))+$")
WORD_PARTS = re.compile(r"[a-z0-9]+")


def read_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def read_pairs(root, split):
    questions = read_lines(root / "data" / f"{split}.nl")
    sql = read_lines(root / "data" / f"{split}.sql")
    return list(zip(questions, sql, strict=True))


def open_database(root):
    uri = (root / "data" / "flight_database.db").resolve().as_uri()
    return sqlite3.connect(uri + "?mode=ro", uri=True)


def phrase_key(text):
    return " ".join(WORD_PARTS.findall(text.casefold()))


class EntitySlots:
    def __init__(self, database):
        choices = defaultdict(set)
        for (city,) in database.execute("SELECT DISTINCT city_name FROM city"):
            if city:
                choices[phrase_key(city)].add(("city", city))
        for code, name in database.execute("SELECT airport_code, airport_name FROM airport"):
            if code and name:
                choices[phrase_key(name)].add(("airport", code))
        for code, name in database.execute("SELECT airline_code, airline_name FROM airline"):
            if code and name:
                full = phrase_key(name)
                for phrase in {full, AIRLINE_SUFFIX.sub("", full)}:
                    if len(phrase) >= 4:
                        choices[phrase].add(("airline", code))
        self.lookup = {
            phrase: next(iter(values))
            for phrase, values in choices.items()
            if phrase and len(values) == 1
        }
        alternatives = (
            r"[\s.\-]+".join(map(re.escape, phrase.split()))
            for phrase in sorted(self.lookup, key=lambda phrase: (-len(phrase), phrase))
        )
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.I)

    def encode_question(self, question):
        reverse = {}

        def replace(match):
            entity = self.lookup[phrase_key(match.group())]
            if entity not in reverse and len(reverse) == 100:
                return match.group()
            if entity not in reverse:
                reverse[entity] = f"<extra_id_{len(reverse)}>"
            return f"{entity[0]} {reverse[entity]}"

        encoded = self.pattern.sub(replace, question)
        return encoded, {slot: value for (_, value), slot in reverse.items()}

    @staticmethod
    def encode_sql(sql, values):
        slots_by_value = {value: slot for slot, value in reversed(list(values.items()))}

        def replace(match):
            value = match.group()[1:-1].replace("''", "'")
            return slots_by_value.get(value, match.group())

        return SQL_LITERAL.sub(replace, sql)

    @staticmethod
    def decode_sql(sql, values):
        def replace(match):
            value = values[match.group()]
            return "'" + value.replace("'", "''") + "'"

        try:
            return SLOT.sub(replace, sql)
        except KeyError:
            return None


def audit(root, slots):
    for split in ("train", "dev"):
        pairs = read_pairs(root, split)
        masked_questions = masked_queries = masked_literals = total_literals = 0
        for question, sql in pairs:
            _, values = slots.encode_question(question)
            encoded = slots.encode_sql(sql, values)
            if slots.decode_sql(encoded, values) != sql:
                raise ValueError(f"SQL transform failed to round-trip in {split}")
            masked_questions += bool(values)
            masked_queries += encoded != sql
            masked_literals += len(SLOT.findall(encoded))
            total_literals += len(SQL_LITERAL.findall(sql))
        print(
            f"{split}: {len(pairs)} rows round-trip exactly | "
            f"{masked_questions} questions with slots | {masked_queries} SQL queries with slots | "
            f"{masked_literals}/{total_literals} quoted literals masked"
        )


def generate_sql(model, tokenizer, slots, questions, database, batch_size, torch):
    from tqdm import trange

    model.eval()
    skip_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    with torch.inference_mode():
        for start in trange(0, len(questions), batch_size, desc="Generating SQL"):
            encoded = [slots.encode_question(text) for text in questions[start:start + batch_size]]
            inputs = tokenizer([text for text, _ in encoded], padding=True, return_tensors="pt")
            outputs = model.generate(
                **inputs.to(model.device), num_beams=4, num_return_sequences=4,
                do_sample=False, max_new_tokens=512,
            ).tolist()
            for index, (_, values) in enumerate(encoded):
                fallback = None
                raw_first = None
                for tokens in outputs[index * 4:(index + 1) * 4]:
                    raw = tokenizer.decode(
                        [token for token in tokens if token not in skip_ids],
                        skip_special_tokens=False, clean_up_tokenization_spaces=False,
                    ).strip()
                    if raw_first is None:
                        raw_first = raw
                    sql = slots.decode_sql(raw, values)
                    if sql is None:
                        continue
                    sql = sql.replace("\r", " ").replace("\n", " ")
                    if fallback is None:
                        fallback = sql
                    try:
                        database.execute("EXPLAIN " + sql).close()
                    except sqlite3.Error:
                        continue
                    yield sql
                    break
                else:
                    yield fallback or (raw_first or "").replace("\r", " ").replace("\n", " ")


def evaluate_dev(root, run, epoch, model, tokenizer, slots, database, args, torch):
    from utils import compute_metrics, save_queries_and_records

    questions = read_lines(root / "data/dev.nl")
    predictions = list(generate_sql(
        model, tokenizer, slots, questions, database, args.eval_batch_size, torch
    ))
    sql_path = run / f"dev_epoch_{epoch}.sql"
    records_path = run / f"dev_epoch_{epoch}.pkl"
    save_queries_and_records(predictions, str(sql_path), str(records_path))
    sql_em, record_em, f1, errors = compute_metrics(
        str(root / "data/dev.sql"), str(sql_path),
        str(root / "records/ground_truth_dev.pkl"), str(records_path),
    )
    print(
        f"Epoch {epoch} | Dev Record F1 {f1:.2%} | Record EM {record_em:.2%} | "
        f"SQL EM {sql_em:.2%} | SQL errors {sum(map(bool, errors))}/{len(errors)}",
        flush=True,
    )
    return float(f1)


def train(root, run, slots, database, args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast

    from load_data import normal_collate_fn
    from t5_utils import initialize_optimizer_and_scheduler
    from train_t5 import train_epoch
    from utils import set_random_seeds

    if not torch.cuda.is_available():
        raise SystemExit("Training requires CUDA. Run this in an HPC GPU terminal.")
    set_random_seeds(42)
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID)
    questions, targets = [], []
    for question, sql in read_pairs(root, "train"):
        encoded, values = slots.encode_question(question)
        questions.append(encoded)
        targets.append(slots.encode_sql(sql, values))
    inputs = tokenizer(questions)["input_ids"]
    labels = tokenizer(targets)["input_ids"]
    loader = DataLoader(
        list(zip(inputs, labels, strict=True)), batch_size=args.batch_size,
        shuffle=True, collate_fn=normal_collate_fn,
    )

    model = T5ForConditionalGeneration(T5Config.from_pretrained(MODEL_ID)).to("cuda")
    options = SimpleNamespace(
        optimizer_type="AdamW", learning_rate=args.learning_rate, weight_decay=0.0,
        scheduler_type="cosine", num_warmup_epochs=1, max_n_epochs=args.epochs,
    )
    optimizer, scheduler = initialize_optimizer_and_scheduler(options, model, len(loader))
    run.mkdir(parents=True, exist_ok=True)
    latest = run / "last.pt"
    completed, best_f1, best_epoch = 0, -1.0, 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["settings"] != vars(args):
            raise SystemExit("Resume with the same training flags, or choose a different --run path.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        completed, best_f1, best_epoch = state["epoch"], state["best_f1"], state["best_epoch"]
        torch.random.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        del state
        print(f"Continuing after epoch {completed}")
    else:
        print("Starting T5-small from randomly initialized weights")

    for epoch in range(completed + 1, args.epochs + 1):
        loss = train_epoch(options, model, loader, optimizer, scheduler)
        print(f"Epoch {epoch}/{args.epochs} | train loss {loss:.4f}", flush=True)
        evaluated = epoch % args.eval_every == 0 or epoch == args.epochs
        if evaluated:
            f1 = evaluate_dev(root, run, epoch, model, tokenizer, slots, database, args, torch)
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                model.save_pretrained(run / "best")
        if not evaluated and epoch % 5 != 0:
            continue
        temporary = latest.with_suffix(".tmp")
        torch.save({
            "settings": vars(args), "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "cpu_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }, temporary)
        temporary.replace(latest)
        print(f"Resume checkpoint saved | Epoch {epoch}", flush=True)
    print(f"Best dev Record F1 {best_f1:.2%} at epoch {best_epoch} | checkpoint {run / 'best'}")


def predict_test(root, run, slots, database, args):
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    from utils import save_queries_and_records

    if not torch.cuda.is_available():
        raise SystemExit("Test prediction requires CUDA. Run this in an HPC GPU terminal.")
    if not (run / "best").is_dir():
        raise SystemExit(f"Missing trained checkpoint: {run / 'best'}")
    model = T5ForConditionalGeneration.from_pretrained(run / "best").to("cuda")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID)
    questions = read_lines(root / "data/test.nl")
    predictions = list(generate_sql(
        model, tokenizer, slots, questions, database, args.eval_batch_size, torch
    ))
    if len(predictions) != len(questions):
        raise RuntimeError("Number of test predictions does not match the test dataset.")
    output = run / "submission"
    output.mkdir(exist_ok=True)
    sql_path = output / "t5_ft_experiment_ec_test.sql"
    records_path = output / "t5_ft_experiment_ec_test.pkl"
    save_queries_and_records(predictions, str(sql_path), str(records_path))
    print(f"Saved {len(predictions)} test predictions to {output}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "train", "predict-test"))
    parser.add_argument("--run", default="checkpoints/scr_experiments/ec_entity_mask")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.eval_every) <= 0:
        parser.error("Epoch and batch counts must be positive.")
    root = Path.cwd()
    if not (root / "data/flight_database.db").is_file():
        parser.error("Run the script from the assignment's part2 directory.")
    with closing(open_database(root)) as database:
        slots = EntitySlots(database)
        if args.mode == "audit":
            audit(root, slots)
        elif args.mode == "predict-test":
            predict_test(root, root / args.run, slots, database, args)
        else:
            train(root, root / args.run, slots, database, args)


if __name__ == "__main__":
    main()
