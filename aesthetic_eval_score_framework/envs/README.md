# Environment Notes

The toolbox keeps one CLI and one JSONL/metrics protocol, but it does not force
all upstream MLLM repositories into one Python environment. In practice, older
official LLaVA/Q-Align code and newer LLaVA-OneVision/Qwen3 code can require
incompatible Transformers versions.

- `requirements-common.txt`: CLI, data loading, metrics, and reporting.
- `requirements-llava37.txt`: ArtiMuse, ArtQuant, AesExpert, UniPercept, OneAlign/Q-Align.
- `requirements-qwen-internvl.txt`: Qwen3VL and InternVL3.5 prompt-numeric adapters.
- `requirements-onevision.txt`: Q-SiT and LLaVA-OneVision 1.5.

Use the same `run.py` commands in every environment; only the Python executable
changes.
