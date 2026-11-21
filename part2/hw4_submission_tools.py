import argparse
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


Q7_CHECKPOINT = Path("checkpoints/ft_experiments/continue_lr5e_5/best")
EC_ORIGINAL = Path("checkpoints/scr_experiments/ec_scratch_to60_lr5e_5/best")
EC_MASKED = Path("checkpoints/scr_experiments/ec_entity_mask/best")

SOURCE_PATHS = {
    "part1": ("main.py", "utils.py", "requirements.txt"),
    "part2": (
        "load_data.py", "train_t5.py", "t5_utils.py", "utils.py", "evaluate.py",
        "data_stats.py", "requirements.txt", "q7_time_repair.py",
        "train_ec_entity_mask.py", "ec_route_hybrid.py", "ec_route_city_repair.py",
    ),
}

ROOT_README = """# DS-GA 1011 Homework 4

`part1/` contains the BERT sentiment-classification code. `part2/` contains
the T5 training, evaluation, and inference code. Use the dependencies listed
in each part's requirements file.

## Reproducing the final Part 2 predictions

Copy the course starter's `part2/data/` into this repository. For development
evaluation, also copy the starter's `part2/records/ground_truth_dev.pkl`.
Extract the checkpoint archive and place its `checkpoints/` folder under
`part2/`, preserving the directory names.

From `part2/`, with the Part 2 environment active on a GPU node:

```bash
python -u hw4_submission_tools.py reproduce-q7
python -u hw4_submission_tools.py reproduce-ec
```

| Submission | Checkpoint directory under part2/ | Final output folder |
| --- | --- | --- |
| Q7 | checkpoints/ft_experiments/continue_lr5e_5/best | submission_q7_time_repair/ |
| Extra Credit, original scratch model (epoch 60) | checkpoints/scr_experiments/ec_scratch_to60_lr5e_5/best | submission_ec_route_grounded/ |
| Extra Credit, entity-slot scratch model (epoch 50) | checkpoints/scr_experiments/ec_entity_mask/best | submission_ec_route_grounded/ |

For Q7, the pipeline generates the initial output with 8-beam search
and a SQLite validity check, then corrects supported explicit time conditions.
For Extra Credit, the original scratch model uses 4-beam search with the same
kind of SQLite check. Simple city-to-city flight questions are routed to
the entity-slot scratch model, then the pipeline corrects supported city-direction conditions.
The GitHub repository holds source code and the checkpoint archive holds trained weights.
Execution timeouts may vary across machines.
"""


def project_root():
    location = Path(__file__).resolve().parent
    for candidate in (location, location.parent):
        if (candidate / "part1/main.py").is_file() and (candidate / "part2/train_t5.py").is_file():
            return candidate
    raise SystemExit("Place this file in the homework repository root or its part2 directory.")


def write_archive(destination, files, compression, readme=None):
    temporary = destination.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=compression) as archive:
        for path, name in files:
            archive.write(path, name)
        if readme is not None:
            archive.writestr("README.md", readme)
    temporary.replace(destination)
    print(f"{destination} | {destination.stat().st_size / 2**20:.1f} MiB")


def pack(root):
    part2 = root / "part2"
    code = []
    for part, names in SOURCE_PATHS.items():
        for name in names:
            path = root / part / name
            if not path.is_file():
                raise SystemExit(f"Missing source file: {path}")
            code.append((path, path.relative_to(root).as_posix()))
    code.append((Path(__file__), "part2/hw4_submission_tools.py"))

    checkpoints = []
    for location in (Q7_CHECKPOINT, EC_ORIGINAL, EC_MASKED):
        directory = part2 / location
        config = directory / "config.json"
        weights = sorted(directory.glob("model*.safetensors"))
        weights += sorted(directory.glob("pytorch_model*.bin"))
        if not config.is_file() or not weights or any(path.stat().st_size == 0 for path in weights):
            raise SystemExit(f"Missing model config or weights: {directory}")
        files = [config, *weights]
        for name in (
            "generation_config.json", "model.safetensors.index.json",
            "pytorch_model.bin.index.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "spiece.model",
        ):
            path = directory / name
            if path.is_file():
                files.append(path)
        checkpoints.extend((path, path.relative_to(part2).as_posix()) for path in files)
        print(f"Checkpoint verified: {location}")

    write_archive(part2 / "hw4_code.zip", code, ZIP_DEFLATED, ROOT_README)
    write_archive(part2 / "hw4_checkpoints.zip", checkpoints, ZIP_STORED)


def generate_initial(checkpoint, output, stem, beams, batch_size, finetune):
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    from load_data import get_dataloader
    from train_t5 import test_inference
    from train_ec_entity_mask import MODEL_ID, open_database

    if not torch.cuda.is_available():
        raise SystemExit("Reproduce predictions on a GPU compute node.")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to("cuda").eval()
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_ID)
    original_generate = model.generate
    output.mkdir(exist_ok=True)

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

    options = SimpleNamespace(finetune=finetune, experiment_name=checkpoint.parent.name)
    with closing(open_database(Path.cwd())) as database:
        model.generate = generate_with_sql_check
        test_inference(
            options, model, get_dataloader(batch_size, "test"),
            str(output / f"{stem}.sql"), str(output / f"{stem}.pkl"),
        )


def reproduce(root, mode):
    part2 = root / "part2"
    os.chdir(part2)
    sys.path.insert(0, str(part2))
    if mode == "reproduce-q7":
        import q7_time_repair

        generate_initial(
            Q7_CHECKPOINT, q7_time_repair.SUBMISSION, q7_time_repair.NAME, 8, 4, True
        )
        q7_time_repair.predict_test(Path.cwd())
    else:
        import ec_route_city_repair
        import ec_route_hybrid
        from train_ec_entity_mask import EntitySlots, open_database

        generate_initial(
            EC_ORIGINAL, ec_route_hybrid.BASELINE_TEST, ec_route_hybrid.TEST_STEM, 4, 8, False
        )
        with closing(open_database(Path.cwd())) as database:
            slots = EntitySlots(database)
            ec_route_hybrid.predict_test(slots, database)
        ec_route_city_repair.predict_test(Path.cwd(), slots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "reproduce-q7", "reproduce-ec"))
    args = parser.parse_args()
    root = project_root()
    if args.mode == "pack":
        pack(root)
    else:
        reproduce(root, args.mode)


if __name__ == "__main__":
    main()
