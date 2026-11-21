from pathlib import Path
from statistics import mean
from transformers import T5TokenizerFast
from load_data import T5Dataset


def print_stats(split, questions, queries, include_count=False):
    print(f"\n{split.upper()}")
    if include_count:
        print(f"Number of examples: {len(questions)}")
    print(f"Mean sentence length: {mean(map(len, questions)):.2f}")
    print(f"Mean SQL query length: {mean(map(len, queries)):.2f}")
    print(f"Vocabulary size (natural language): {len({t for ids in questions for t in ids})}")
    print(f"Vocabulary size (SQL): {len({t for ids in queries for t in ids})}")


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    splits = ("train", "dev")

    print("TABLE 1 - Before preprocessing")
    for split in splits:
        questions = (data_dir / f"{split}.nl").read_text(encoding="utf-8").splitlines()
        queries = (data_dir / f"{split}.sql").read_text(encoding="utf-8").splitlines()
        if len(questions) != len(queries):
            raise ValueError(f"Mismatched questions and SQL for {split}")

        question_ids = tokenizer(questions, add_special_tokens=False)["input_ids"]
        query_ids = tokenizer(queries, add_special_tokens=False)["input_ids"]
        print_stats(split, question_ids, query_ids, include_count=True)

    print("\nTABLE 2 - After preprocessing (google-t5/t5-small)")
    for split in splits:
        dataset = T5Dataset(data_dir, split)
        question_ids, query_ids = zip(*dataset.examples)
        print_stats(split, question_ids, query_ids)


if __name__ == "__main__":
    main()
