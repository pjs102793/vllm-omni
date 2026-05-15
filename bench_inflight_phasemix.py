"""Simulate scheduler-integrated in-flight: a single forward where
each row has a *different* cache_len (= different sub-step phase).

This is the throughput scenario unlocked by a per-request scheduler in
gpu_model_runner. The forward kernel can run it today; only the
calling architecture needs to change. This bench measures whether the
kernel itself wins anything from phase diversity vs the current
lock-step (KV-graph decode at fixed cache_len).
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from vllm_omni.model_executor.models.common.qwen3_code_predictor_inflight import (
    CodePredictorBaseModelInflight,
)
from vllm_omni.model_executor.models.common.qwen3_code_predictor_kv_graph import (
    CodePredictorBaseModelKVGraph,
)


class CPCfg:
    num_code_groups = 16
    hidden_size = 1024
    intermediate_size = 3072
    num_attention_heads = 16
    num_key_value_heads = 8
    num_hidden_layers = 5
    head_dim = 128
    rms_norm_eps = 1e-6
    rope_theta = 1_000_000
    attention_bias = False
    vocab_size = 4096
    max_position_embeddings = 65536


def time_graph(g, iters: int, trials: int = 3) -> float:
    medians = []
    for _ in range(trials):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            g.replay()
            ends[i].record()
        torch.cuda.synchronize()
        ts = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
        medians.append(statistics.median(ts))
        time.sleep(0.05)
    return statistics.median(medians)


def measure_inflight_phasemix(model: CodePredictorBaseModelInflight, N: int, dtype):
    """Run inflight forward at N tokens with cache_lens spread across phases.

    cache_lens distribution: round-robin across [0, 1, ..., Q-1].
    """
    device = next(model.parameters()).device
    state = model.allocate_state(N, device, dtype)
    Q = model.config.num_code_groups
    max_seq = Q + 1

    inputs = torch.randn(N, 1, model.config.hidden_size, device=device, dtype=dtype)

    slot_indices = torch.arange(N, device=device, dtype=torch.long)
    cache_lens = torch.tensor([i % (max_seq - 1) for i in range(N)], device=device, dtype=torch.long)
    write_positions = cache_lens
    position_ids = cache_lens.unsqueeze(-1)
    attn_mask = torch.zeros(N, 1, 1, max_seq, device=device, dtype=torch.bool)
    ar = torch.arange(max_seq, device=device).unsqueeze(0)
    attn_mask[:, 0, 0, :] = ar <= cache_lens.unsqueeze(1)

    fwd = torch.compile(model.forward, dynamic=False, options={"epilogue_fusion": False})
    for _ in range(20):
        _ = fwd(inputs, position_ids, slot_indices, write_positions, attn_mask,
                state.k_caches, state.v_caches)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fwd(inputs, position_ids, slot_indices, write_positions, attn_mask,
                  state.k_caches, state.v_caches)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    return time_graph(g, iters=500)


def measure_inflight_lockstep(model: CodePredictorBaseModelInflight, N: int, dtype, cache_len: int = 7):
    """Inflight forward at N tokens, all rows at the same cache_len (== current PoC)."""
    device = next(model.parameters()).device
    state = model.allocate_state(N, device, dtype)
    Q = model.config.num_code_groups
    max_seq = Q + 1

    inputs = torch.randn(N, 1, model.config.hidden_size, device=device, dtype=dtype)

    slot_indices = torch.arange(N, device=device, dtype=torch.long)
    cache_lens = torch.full((N,), cache_len, device=device, dtype=torch.long)
    write_positions = cache_lens
    position_ids = cache_lens.unsqueeze(-1)
    attn_mask = torch.zeros(N, 1, 1, max_seq, device=device, dtype=torch.bool)
    attn_mask[:, 0, 0, : cache_len + 1] = True

    fwd = torch.compile(model.forward, dynamic=False, options={"epilogue_fusion": False})
    for _ in range(20):
        _ = fwd(inputs, position_ids, slot_indices, write_positions, attn_mask,
                state.k_caches, state.v_caches)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fwd(inputs, position_ids, slot_indices, write_positions, attn_mask,
                  state.k_caches, state.v_caches)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    return time_graph(g, iters=500)


def measure_kvgraph_decode(model: CodePredictorBaseModelKVGraph, N: int, dtype, cache_len: int = 7):
    device = next(model.parameters()).device
    state = model.allocate_state(N, device, dtype)
    state.reset()
    state.set_decode_step(cache_len)
    dec_in = torch.randn(N, 1, model.config.hidden_size, device=device, dtype=dtype)
    dec_pos = torch.tensor([[cache_len]], device=device, dtype=torch.long).expand(N, -1).contiguous()

    fwd = torch.compile(model.forward_decode, dynamic=False, options={"epilogue_fusion": False})
    for _ in range(20):
        _ = fwd(dec_in, dec_pos, state.k_caches, state.v_caches, state.write_idx, state.attn_mask)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fwd(dec_in, dec_pos, state.k_caches, state.v_caches, state.write_idx, state.attn_mask)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    return time_graph(g, iters=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default="8,16,32,64,128,256", help="comma N values")
    args = ap.parse_args()

    cfg = CPCfg()
    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n{'N':>4} | {'inflight phasemix (us)':>22} | {'inflight lockstep (us)':>22} | {'kvgraph lockstep (us)':>22}")
    print("-" * 90)
    for N in [int(x) for x in args.N.split(",")]:
        torch.manual_seed(0)
        infl = CodePredictorBaseModelInflight(cfg, embedding_dim=cfg.hidden_size).to(device=device, dtype=dtype).eval()
        torch.manual_seed(0)
        kvg = CodePredictorBaseModelKVGraph(cfg, embedding_dim=cfg.hidden_size).to(device=device, dtype=dtype).eval()

        try:
            t_pm = measure_inflight_phasemix(infl, N, dtype)
            t_ls = measure_inflight_lockstep(infl, N, dtype)
            t_kv = measure_kvgraph_decode(kvg, N, dtype)
        except Exception as e:
            print(f"{N:>4} | failed: {e}")
            continue

        print(f"{N:>4} | {t_pm:>19.1f}    | {t_ls:>19.1f}    | {t_kv:>19.1f}")

        del infl, kvg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
