# DS-GA 1011 Homework 4

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
