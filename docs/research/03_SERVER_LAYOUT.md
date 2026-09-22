# Server artifact layout (public-safe description)

Actual connection details and absolute private paths live in the ignored local file `.codex-local/SERVER_PATHS.md`.

Expected layout under the server repository:

```text
inter_hierarchy_MN40/
├── checkpoints/                     # historical pretrained checkpoints
├── runs_v2/
│   ├── checkpoint_eval_20260922/
│   ├── control_beta0_s22_e20/
│   ├── main_exact_lca_s22_e20/      # stopped cosine gate
│   ├── main_hypdist_exact_lca_s22_e20/
│   └── struct_*.json
├── main_inter_v2.py
├── evaluate_structure_v2.py
└── v2/
```

The ModelNet40 HDF5 dataset is server-local. `runs_v2/`, `*.pth`, `*.pt`, logs, and dataset directories are ignored by Git.

Existing result identities:

- zero-increment control: `control_beta0_s22_e20`;
- selected first main run: `main_hypdist_exact_lca_s22_e20`;
- failed cosine gate retained for evidence: `main_exact_lca_s22_e20`.

