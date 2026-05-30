# <h1 align="center">*<ins>Notebooks</ins>*</h1>

This directory contains the analysis notebooks used to inspect architecture ablations, quantization experiments, and the final combined model.

The notebooks mainly serve for:

- plotting validation BPB and compressed model size;
- comparing single-seed sweeps and multi-seed validations;
- analysing architectural mechanisms;
- visualising quantization trade-offs;
- preparing figures and tables for the final report.

## Directory structure

```text
notebooks/
├── architecture_notebooks/
│   ├── ablation3_parallel_residuals.ipynb
│   ├── ablation4_attention_residuals.ipynb
│   ├── ablation5_depth_recurrence_v2.ipynb
│   ├── ablation7_attn_gate.ipynb
│   ├── ablation8_xsa_analysis.ipynb
│   ├── ablations1_2_optimizer.ipynb
│   └── utils.py
├── final_model_notebooks/
│   ├── final_model_combined_ablation.ipynb
│   └── utils.py
└── quantization_notebooks/
    ├── ablation_gptq_lqer.ipynb
    ├── ablation_gptq_vs_naive.ipynb
    ├── ablation_nf4_analysis.ipynb
    ├── ablation_quant_2d_matrices.ipynb
    ├── ablation_quant_all_tensors.ipynb
    ├── ablation_quant_design_study_analysis.ipynb
    ├── awq_quantization.ipynb
    ├── layer_sensitivity_quantization.ipynb
    └── utils.py
```

## How to use these notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Some notebooks may expect local experiment logs, Weights & Biases exports, or saved CSV/JSON summaries. These generated artifacts are not necessarily included in the repository.

## Notes

The notebooks are analysis-first: they are meant to explain, compare, and visualise experiments. The training code itself lives in `src/`.