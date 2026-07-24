"""
Sample from a trained model
"""
import os
import pickle
import hashlib
import json
import platform
import statistics
import threading
import time
from contextlib import nullcontext

import psutil
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
inference_backend = 'torch' # 'torch', 'rns', or 'software-rns'
rns_quant_bits = 8
rns_attention = True # use RNS for QK and attention-value matmul too
rns_include_lm_head = True
rns_fused = True
rns_lut_channels = 2
benchmark = False
benchmark_warmup = 1
benchmark_runs = 3
benchmark_output = '' # auto: <out_dir>/inference_<backend>.json
print_samples = True
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

if inference_backend not in ('torch', 'rns', 'software-rns'):
    raise ValueError("inference_backend must be 'torch', 'rns', or 'software-rns'")
if benchmark_warmup < 0 or benchmark_runs < 1:
    raise ValueError("benchmark_warmup must be >= 0 and benchmark_runs must be >= 1")

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
def inference_context():
    return nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
rns_installation = None
if inference_backend != 'torch':
    if compile:
        raise ValueError("torch.compile is not supported with RNS inference")
    if inference_backend == 'rns' and device_type != 'cuda':
        raise ValueError("CUDA RNS inference requires --device=cuda")
    from rns_inference import install_rns_inference
    rns_installation = install_rns_inference(
        model,
        mode=inference_backend,
        quant_bits=rns_quant_bits,
        include_attention_matmul=rns_attention,
        include_lm_head=rns_include_lm_head,
        fused=rns_fused,
        lut_channels=rns_lut_channels,
    )
    print(
        f"Installed {inference_backend}: "
        f"{len(rns_installation.replaced_linears)} Linear modules, "
        f"{rns_installation.attention_blocks} attention blocks"
    )
    print(f"RNS package: {rns_installation.package_path}")
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

def synchronize():
    if device_type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def generate_once(run_seed):
    torch.manual_seed(run_seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(run_seed)
    synchronize()
    started = time.perf_counter()
    with inference_context():
        result = model.generate(
            x.clone(),
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_benchmark(result, generated):
    output_path = benchmark_output or os.path.join(
        out_dir, f"inference_{inference_backend}.json"
    )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    log_path = os.path.splitext(output_path)[0] + '.log'
    summary = [
        f"backend={inference_backend}",
        f"device={result['hardware']['device']}",
        f"generated_tokens={max_new_tokens}",
        f"latency_p50_ms={result['timing']['latency_p50_ms']:.3f}",
        f"latency_p95_ms={result['timing']['latency_p95_ms']:.3f}",
        f"tokens_per_second={result['timing']['tokens_per_second']:.3f}",
        f"gpu_peak_allocated_bytes={result['memory']['gpu_peak_allocated_bytes']}",
        f"gpu_peak_reserved_bytes={result['memory']['gpu_peak_reserved_bytes']}",
        f"process_peak_rss_bytes={result['memory']['process_peak_rss_bytes']}",
        f"backend_stats={result['backend_stats']}",
        f"json={output_path}",
    ]
    with open(log_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(summary) + '\n')
    print('\n'.join(summary))
    if print_samples:
        print(decode(generated[0].tolist()))
    print(f"log={log_path}")


if benchmark:
    warmup_ms = []
    y = None
    for warmup_index in range(benchmark_warmup):
        y, elapsed = generate_once(seed)
        warmup_ms.append(elapsed)

    if rns_installation is not None:
        rns_installation.reset_stats()
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop_monitor = threading.Event()

    def monitor_rss():
        while not stop_monitor.wait(0.01):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor = threading.Thread(target=monitor_rss, daemon=True)
    monitor.start()
    measured_ms = []
    try:
        for run_index in range(benchmark_runs):
            y, elapsed = generate_once(seed)
            measured_ms.append(elapsed)
    finally:
        stop_monitor.set()
        monitor.join()
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    median_ms = statistics.median(measured_ms)
    generated_ids = y[0].tolist()
    token_bytes = ','.join(map(str, generated_ids)).encode('ascii')
    cuda_available = device_type == 'cuda'
    result = {
        "backend": inference_backend,
        "timestamp_unix": time.time(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(0) if cuda_available else str(device),
            "device_type": device_type,
        },
        "model": {
            "out_dir": os.path.abspath(out_dir),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "block_size": model.config.block_size,
            "prompt_tokens": len(start_ids),
            "generated_tokens": max_new_tokens,
            "dtype": dtype,
            "compile": compile,
        },
        "rns": {
            "quant_bits": rns_quant_bits,
            "attention_matmul": rns_attention,
            "include_lm_head": rns_include_lm_head,
            "fused": rns_fused,
            "lut_channels": rns_lut_channels,
            "installation": (
                rns_installation.metadata() if rns_installation is not None else None
            ),
        },
        "timing": {
            "synchronization": "torch.cuda.synchronize" if cuda_available else "none",
            "warmup_runs": benchmark_warmup,
            "warmup_ms": warmup_ms,
            "measured_runs": benchmark_runs,
            "samples_ms": measured_ms,
            "latency_p50_ms": median_ms,
            "latency_p95_ms": percentile(measured_ms, 0.95),
            "tokens_per_second": max_new_tokens / (median_ms / 1000.0),
        },
        "memory": {
            "gpu_allocated_bytes": torch.cuda.memory_allocated() if cuda_available else 0,
            "gpu_reserved_bytes": torch.cuda.memory_reserved() if cuda_available else 0,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda_available else 0,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda_available else 0,
            "process_peak_rss_bytes": peak_rss[0],
        },
        "backend_stats": (
            rns_installation.stats_snapshot() if rns_installation is not None else {}
        ),
        "output": {
            "token_ids": generated_ids,
            "sha256": hashlib.sha256(token_bytes).hexdigest(),
        },
    }
    write_benchmark(result, y)
else:
    for sample_index in range(num_samples):
        y, _ = generate_once(seed + sample_index)
        print(decode(y[0].tolist()))
        print('---------------')
