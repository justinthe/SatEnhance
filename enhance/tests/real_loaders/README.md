# Real model loader scripts (test fixtures)

Each SEN2SR model on Hugging Face ships a `load.py` that `mlstac` downloads and `exec`s when the
model is loaded. Our tests run the **real** scripts so that missing packages (for example
`matplotlib`, which every one of them imports) show up in tests rather than on the first real run.

| File | Source (`hf://models/tacofoundation/sen2sr/...`) |
|---|---|
| `lite_rgbn_x4_load.py` | `SEN2SRLite/NonReference_RGBN_x4/load.py` |
| `lite_main_load.py` | `SEN2SRLite/main/load.py` |
| `full_rgbn_x4_load.py` | `SEN2SR/NonReference_RGBN_x4/load.py` |
| `full_main_load_header.py` | import lines of `SEN2SR/main/load.py` (the body needs `mamba_ssm` and a GPU) |

Copied on 2026-09-30. The linter normalised import order and trailing whitespace in the copies;
the code is otherwise as published. They are excluded from linting (ruff.toml) so they stay
as they are. Refresh them from the repository if SEN2SR changes its loaders.
