# Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 Benchmark Instructions

Install the pinned dependency versions from `requirements/requirements_nemotron_omni.txt`.

Launch `./run_server.sh` in a separate terminal to start the vLLM server. 
The script has a few options that can be configured (e.g., disable torch.compile, or flags for certain GPUs).

Once the server is ready, run `./run_nemotron_omni_vllm.sh`.
