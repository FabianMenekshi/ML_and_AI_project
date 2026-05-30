# <h1 align="center">*<ins>Source Code</ins>*</h1>

This directory contains the training and experiment scripts for the project.

The code is organised into three main groups:

```text
src/
├── architecture/
├── quantization/
└── final_model/
```

- `architecture/` contains architectural ablations and composition experiments.
- `quantization/` contains post-training and training-aware quantization experiments.
- `final_model/` contains the final combined model script.

Most scripts are self-contained training files derived from the baseline GPT script. They define the model, optimizer, training loop, validation, quantization, and experiment-specific modifications in one file.

## Common assumptions

Most scripts expect:

- pre-tokenized FineWeb training and validation shards;
- a SentencePiece tokenizer;
- a CUDA-capable GPU;
- PyTorch;
- Weights & Biases for logging, if enabled.

## Logging

All the scripts support Weights & Biases logging through `wandb`.

To enable logging:

```bash
wandb login
export WANDB_PROJECT=parameter-golf-ablation-study
```

## SLURM scripts

The SLURM `.sh` launch scripts used to run the experiments on the cluster are not included on the `main` branch. They are kept on the corresponding individual experiment branches, together with the exact run commands and branch-specific settings used for those experiments.

## Notes for contributors

- Scripts are intentionally self-contained for reproducibility.
- Many files duplicate baseline model code with only the experiment-specific mechanism changed.
- Environment variables are preferred over hard-coded experiment settings.
- Generated data, checkpoints, logs, and compressed artifacts should generally not be committed.