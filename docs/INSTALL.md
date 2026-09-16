# Installation

## Verified runtime

The successful server runs used `/dataB/zmm/conda_envs/zmm/bin/python` with Python 3.10.18. Critical versions were:

| Package | Version |
|---|---|
| PyTorch | 2.4.0+cu121 |
| torchvision | 0.19.0+cu121 |
| transformers | 4.49.0 |
| PEFT | 0.19.1 |
| bitsandbytes | 0.45.5 |
| accelerate | 0.34.2 |
| anyio | 4.13.0 |
| MCP Python SDK | 1.29.0 |
| Triton | 3.0.0 |
| datasets | 2.21.0 |
| pyarrow | 15.0.2 |

Create a clean equivalent environment:

```bash
conda env create -f environment/environment.yml
conda activate multimodal-web-agent
python -m pip install -e . --no-deps
```

On the original server, use the verified interpreter without changing the environment:

```bash
export MWA_ROOT="$PWD"
export MWA_PYTHON=/dataB/zmm/conda_envs/zmm/bin/python
export PYTHONPATH="$MWA_ROOT/src:$PYTHONPATH"
$MWA_PYTHON -c 'import torch, transformers, peft, bitsandbytes; print(torch.__version__)'
```

Training requires a CUDA 12.1-compatible NVIDIA driver. O1 Live evaluation requires `SERPER_API_KEY` and `SERPAPI_API_KEY`; fresh E-VQA text-Web stages require `DASHSCOPE_API_KEY`. Keep all credentials in environment variables or a local ignored `.env`, never in Git.
