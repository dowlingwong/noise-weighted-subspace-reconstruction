# NPML Talk Notebooks

Generated notebooks live directly in this folder. This is the former root
`NPML/` workspace, now kept with the Paper 2 reconstruction stack.

Outputs used by the notebooks:

- figures: `paper2/npml/figures/`
- tables: `paper2/npml/tables/`

Regenerate the notebook set and outputs from the repository root with:

```bash
PYTHONPATH=. python paper2/npml/generate_npml_notebooks.py
```

Notebook set:

- `00_model_inventory_and_gap_map.ipynb`
- `01_experiment_a_metric_ablation.ipynb`
- `02_experiment_b_coverage_ablation.ipynb`
- `03_experiment_c_nfpa_vs_empca.ipynb`
- `04_experiment_d_architecture_bias_readiness.ipynb`
- `05_experiment_e_prewhitened_transformer_readiness.ipynb`
- `06_final_2x2_matrix.ipynb`
