# chronogpt_lmeval.py
from __future__ import annotations

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"  # allow legacy dataset scripts (e.g. social_iqa)


import gc
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from lm_eval.api.model import TemplateLM
from lm_eval import utils as lm_utils  # get_rolling_token_windows, make_disjoint_window
import tiktoken
from tqdm import tqdm

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    @torch.inference_mode()
    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum('i,j -> ij', t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    @torch.inference_mode()
    def forward(self, x):
        cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = CastedLinear(dim, dim)
        self.register_buffer('kv_cache', None, persistent=False)

    @torch.inference_mode()
    def forward(self, x, ve):
        B, T = x.size(0), x.size(1)
        
        # Generate Q, K, V
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
            
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        
        # Use KV cache if available
        if self.kv_cache is not None:
            k = torch.cat([self.kv_cache[0], k], dim=1)
            v = torch.cat([self.kv_cache[1], v], dim=1)
            self.kv_cache = torch.stack([k, v])

        # Efficient attention with flash attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),  # (B, num_heads, T, head_dim)
                k.transpose(1, 2),  # (B, num_heads, T, head_dim)
                v.transpose(1, 2),  # (B, num_heads, T, head_dim)
                is_causal=True
            )
        else:
            # Fallback to regular attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(
                torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(),
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    @torch.inference_mode()
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, model_dim, num_heads, use_attn=True):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads) if use_attn else None
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    @torch.inference_mode()
    def forward(self, x, ve, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers=52):
        super().__init__()
        self.num_layers = num_layers
        # We only have 3 distinct embedding modules, reused at beginning and end.
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs):
        # Compute the base embeddings (a list of length 3)
        base = [emb(inputs).bfloat16() for emb in self.embed]
        L = self.num_layers
        half = L // 2  # number of encoder layers (assumes num_layers is even)
        # Build encoder: first 3 layers get embeddings, rest get None.
        encoder = [base[i] if i < 3 else None for i in range(half)]
        # Build decoder: last 3 layers get embeddings, others get None.
        # For decoder layers, if i is in [half-3, half-1] then assign base[0], base[1], base[2]
        decoder = [base[i - (half - 3)] if i >= (half - 3) else None for i in range(half)]
        return encoder + decoder


class ChronoGPT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size  # Store vocab_size as instance variable
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, use_attn=True) for i in range(num_layers)])
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers=num_layers)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
    @torch.inference_mode()
    def forward(self, inputs, past_key_values=None):
        # Remove fixed batch size assumption
        B = inputs.size(0)  # Get batch size from input tensor
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # Add batch dimension if not present
        
        x0 = norm(self.embed(inputs).bfloat16())
        x = x0
        
        # Modify value embedding handling for batched input
        ve = [self.value_embeds(inputs[i].view(-1)) for i in range(B)]
        ve = [torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None 
              for i in range(len(ve[0]))]
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Handle cached states for batched input
        if past_key_values is not None:
            for i, block in enumerate(self.blocks):
                if block.attn is not None:
                    block.attn.kv_cache = past_key_values[i]

        present = []
        layer_outputs = []
        skip_connections = []

        # Process through encoder layers
        for i in range(self.num_encoder_layers):
            block = self.blocks[i]
            x = block(x, ve_enc[i], x0)
            if block.attn is not None:
                present.append(block.attn.kv_cache)
                block.attn.kv_cache = None
            skip_connections.append(x)
            layer_outputs.append(norm(x))

        # Process through decoder layers
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            block = self.blocks[self.num_encoder_layers + i]
            x = block(x, ve_dec[i], x0)
            layer_outputs.append(norm(x))
            if block.attn is not None:
                present.append(block.attn.kv_cache)
                block.attn.kv_cache = None

        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)

        return logits.float(), layer_outputs
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        config = {
            "model_type": "ChronoGPT",
            "vocab_size": self.embed.num_embeddings,
            "num_layers": len(self.blocks),
            "num_heads": self.num_heads,
            "model_dim": self.embed.embedding_dim
        }
        torch.save(config, os.path.join(save_directory, "config.pt"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)
    @classmethod
    def from_pretrained(cls, repo_id, cache_dir=None, **kwargs):
        config_path = hf_hub_download(repo_id=repo_id, filename="config.pt", cache_dir=cache_dir)
        bin_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=cache_dir)
        config = torch.load(config_path)
        model = cls(**config)
        model.load_state_dict(torch.load(bin_path))
        return model


class ChronoGPTLM(TemplateLM):
    """
    Minimal lm-eval-harness adapter for ChronoGPT (custom PyTorch model + tiktoken).
    Implements:
      - _loglikelihood_tokens (used by TemplateLM.loglikelihood)
      - loglikelihood_rolling
      - generate_until
    """

    backend = "causal"

    def __init__(
        self,
        repo_id: str = "manelalab/chrono-gpt-v1-20241231",
        cache_dir: str = "cache",
        device: str | None = "cuda",
        dtype: str = "float16",
        batch_size: int = 1,
        max_length: int | None = None,
        max_gen_toks: int = 256,
        seed: int = 1234,
        vocab_size: int | None = None,
    ):
        super().__init__()

        # ---- device ----
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._device = torch.device(device)

        # ---- tokenizer ----
        self._tok = tiktoken.get_encoding("gpt2")

        # gpt2 EOT is 50256; try to get it robustly if tiktoken supports it
        self._eot_token_id = 50256
        try:
            self._eot_token_id = self._tok.encode(
                "<|endoftext|>", allowed_special={"<|endoftext|>"}
            )[0]
        except Exception:
            pass

        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)

        # ---- load config + weights from HF Hub ----
        config_path = hf_hub_download(repo_id=repo_id, filename="config.pt", cache_dir=cache_dir)
        weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=cache_dir)

        config = torch.load(config_path, map_location="cpu")
        self._config = config

        model = ChronoGPT(**config).to(self._device).eval()

        if dtype.lower() in ("float16", "fp16", "half"):
            model = model.half()
        elif dtype.lower() in ("bfloat16", "bf16"):
            model = model.bfloat16()
        elif dtype.lower() in ("float32", "fp32", "float"):
            pass
        else:
            raise ValueError(f"Unsupported dtype={dtype!r}")

        state_dict = torch.load(weights_path, map_location=self._device)
        model.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache() if self._device.type == "cuda" else None
        gc.collect()

        self._model = model

        # ---- max_length (context length) ----
        # Try common names, else default.
        if max_length is None:
            max_length = (
                config.get("block_size")
                or config.get("n_ctx")
                or config.get("max_seq_len")
                or config.get("max_length")
                or 2048
            )
        self._max_length = int(max_length)

        # ---- vocab size ----
        self.vocab_size = int(vocab_size or config.get("vocab_size") or 50257)

        # ---- RNG for sampling ----
        self._gen = torch.Generator(device=self._device)
        self._gen.manual_seed(int(seed))

    # ---------------- lm-eval required properties ----------------

    @property
    def eot_token_id(self) -> int:
        return int(self._eot_token_id)

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def tokenizer_name(self) -> str:
        # Used for cache fingerprinting in some setups; safe to provide.
        return "tiktoken_gpt2"

    # ---------------- tokenization ----------------

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> List[int]:
        # allow_special="all" avoids crashes if special tokens appear
        return self._tok.encode(string, allowed_special="all")

    def tok_decode(self, tokens: List[int]) -> str:
        return self._tok.decode(tokens)

    # ---------------- core scoring ----------------

    def _model_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T] long
        returns logits: [B, T, V]
        """
        out = self._model(input_ids)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        return logits

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """
        requests: [ ((context_str, cont_str), context_enc, continuation_enc), ... ]
        returns:  [ (logprob_sum, is_greedy), ... ]
        """
        results: List[Tuple[float, bool]] = []
        bs = max(1, int(self.batch_size))

        # simple batching (right-pad is safe for causal LMs)
        n_batches = (len(requests) + bs - 1) // bs
        pbar = tqdm(total=len(requests), desc="loglikelihood", disable=disable_tqdm)
        for start in range(0, len(requests), bs):
            chunk = requests[start : start + bs]

            input_tok_lists: List[List[int]] = []
            cont_tok_lists: List[List[int]] = []
            input_lens: List[int] = []

            for (_ctx_str, _cont_str), ctx_enc, cont_enc in chunk:
                # Empty continuation => zero score.
                if len(cont_enc) == 0:
                    input_tok_lists.append([self.prefix_token_id])
                    cont_tok_lists.append([])
                    input_lens.append(1)
                    continue

                # Truncate to fit model context.
                # We feed: input_tokens = ctx + cont[:-1]
                # So constraint is: len(ctx) + len(cont) - 1 <= max_length
                max_seq = self.max_length

                if len(cont_enc) > max_seq:
                    # Can't score more than max_seq tokens with ctx_len>=1
                    cont_enc = cont_enc[-max_seq:]

                max_ctx = max_seq - len(cont_enc) + 1
                if len(ctx_enc) > max_ctx:
                    ctx_enc = ctx_enc[-max_ctx:]

                input_tokens = ctx_enc + cont_enc[:-1]
                input_tok_lists.append(input_tokens)
                cont_tok_lists.append(cont_enc)
                input_lens.append(len(input_tokens))

            max_inp_len = max(input_lens)
            pad_id = self.eot_token_id

            input_ids = torch.full(
                (len(chunk), max_inp_len),
                fill_value=pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for i, toks in enumerate(input_tok_lists):
                input_ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=self.device)

            with torch.inference_mode():
                logits = self._model_forward(input_ids)  # [B, T, V]

            log_probs = F.log_softmax(logits.float(), dim=-1)

            for i in range(len(chunk)):
                cont_ids = cont_tok_lists[i]
                if len(cont_ids) == 0:
                    results.append((0.0, True))
                    continue

                inp_len = input_lens[i]
                k = len(cont_ids)

                # Score the last k logits from the *unpadded* portion
                lp_slice = log_probs[i, inp_len - k : inp_len, :]  # [k, V]

                cont_t = torch.tensor(cont_ids, dtype=torch.long, device=self.device)  # [k]
                token_lp = lp_slice.gather(dim=-1, index=cont_t.unsqueeze(-1)).squeeze(-1)  # [k]
                ll = float(token_lp.sum().item())

                greedy_ids = lp_slice.argmax(dim=-1)
                is_greedy = bool((greedy_ids == cont_t).all().item())

                results.append((ll, is_greedy))

            pbar.update(len(chunk))
        pbar.close()
        return results

    # ---------------- rolling loglikelihood (perplexity) ----------------

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        """
        Uses lm_eval.utils.get_rolling_token_windows + make_disjoint_window to chunk
        long strings into max_length windows. :contentReference[oaicite:3]{index=3}
        """
        out: List[float] = []
        for (string,) in tqdm([req.args for req in requests], desc="rolling_ll", disable=disable_tqdm):
            token_list = self.tok_encode(string)

            windows = list(
                map(
                    lm_utils.make_disjoint_window,
                    lm_utils.get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # windows: List[(context_tokens, continuation_tokens)]
            tok_reqs = [(("", ""), ctx, cont) for (ctx, cont) in windows]
            lls = self._loglikelihood_tokens(tok_reqs, disable_tqdm=True)
            total_ll = sum(ll for ll, _ in lls)
            out.append(float(total_ll))

        return out

    # ---------------- generation ----------------

    def _sample_next(
        self,
        logits_1d: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
    ) -> int:
        # Greedy by default
        if (not do_sample) or temperature == 0.0:
            return int(torch.argmax(logits_1d).item())

        x = logits_1d.float() / max(temperature, 1e-8)

        if top_k is not None and top_k > 0:
            k = min(int(top_k), x.numel())
            vals, idx = torch.topk(x, k)
            probs = F.softmax(vals, dim=-1)
            pick = torch.multinomial(probs, num_samples=1, generator=self._gen).item()
            return int(idx[pick].item())

        if top_p is not None and 0.0 < float(top_p) < 1.0:
            sorted_logits, sorted_idx = torch.sort(x, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            mask = cum > float(top_p)
            mask[0] = False  # keep at least one
            sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
            probs = F.softmax(sorted_logits, dim=-1)
            pick = torch.multinomial(probs, num_samples=1, generator=self._gen).item()
            return int(sorted_idx[pick].item())

        probs = F.softmax(x, dim=-1)
        return int(torch.multinomial(probs, num_samples=1, generator=self._gen).item())

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """
        Batched (right-padded) autoregressive generation.
        Returns ONLY the continuation text (not including the prompt).
        """
        outs: List[str] = []
        bs = max(1, int(self.batch_size))

        pbar = tqdm(total=len(requests), desc="generate", disable=disable_tqdm)
        for start in range(0, len(requests), bs):
            chunk = requests[start : start + bs]

            contexts: List[str] = []
            gen_kwargs_list: List[Dict[str, Any]] = []

            for req in chunk:
                context, gen_kwargs = req.args
                contexts.append(context)
                gen_kwargs_list.append(gen_kwargs or {})

            # Per-request generation params
            untils: List[List[str]] = []
            max_news: List[int] = []
            temps: List[float] = []
            topks: List[int | None] = []
            topps: List[float | None] = []
            do_samples: List[bool] = []

            for g in gen_kwargs_list:
                u = g.get("until") or g.get("stop") or []
                if isinstance(u, str):
                    u = [u]
                untils.append(list(u))

                max_new = g.get("max_gen_toks", g.get("max_new_tokens", g.get("max_tokens", self.max_gen_toks)))
                max_news.append(int(max_new))

                t = float(g.get("temperature", 0.0))
                temps.append(t)

                tk = g.get("top_k", None)
                topks.append(int(tk) if tk is not None else None)

                tp = g.get("top_p", None)
                topps.append(float(tp) if tp is not None else None)

                # default behavior: sample if temperature > 0 unless do_sample explicitly provided
                if "do_sample" in g:
                    do_samples.append(bool(g["do_sample"]))
                else:
                    do_samples.append(t > 0.0)

            # Tokenize prompts
            seqs: List[List[int]] = [self.tok_encode(c) for c in contexts]
            for i, s in enumerate(seqs):
                if len(s) == 0:
                    seqs[i] = [self.prefix_token_id]

            gen_tokens: List[List[int]] = [[] for _ in seqs]
            finished = [False] * len(seqs)
            final_text: List[str | None] = [None] * len(seqs)

            max_steps = max(max_news) if max_news else 0

            for step in range(max_steps):
                if all(finished):
                    break

                lengths = [len(s) for s in seqs]
                max_len = max(lengths)
                pad_id = self.eot_token_id

                input_ids = torch.full(
                    (len(seqs), max_len),
                    fill_value=pad_id,
                    dtype=torch.long,
                    device=self.device,
                )
                for i, s in enumerate(seqs):
                    input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)

                with torch.inference_mode():
                    logits = self._model_forward(input_ids)  # [B, T, V]

                for i in range(len(seqs)):
                    if finished[i]:
                        continue
                    if step >= max_news[i]:
                        # reached token budget
                        txt = self.tok_decode(gen_tokens[i])
                        final_text[i] = _truncate_at_any(txt, untils[i])
                        finished[i] = True
                        continue

                    last_pos = lengths[i] - 1
                    next_logits = logits[i, last_pos, :]

                    next_id = self._sample_next(
                        next_logits,
                        do_sample=do_samples[i],
                        temperature=temps[i],
                        top_k=topks[i],
                        top_p=topps[i],
                    )

                    seqs[i].append(next_id)
                    gen_tokens[i].append(next_id)

                    # Stop on EOT
                    if next_id == self.eot_token_id:
                        txt = self.tok_decode(gen_tokens[i])
                        final_text[i] = _truncate_at_any(txt, untils[i])
                        finished[i] = True
                        continue

                    # Stop sequences (string-based)
                    if untils[i]:
                        txt = self.tok_decode(gen_tokens[i])
                        cut_txt = _truncate_at_any(txt, untils[i])
                        if cut_txt != txt:
                            final_text[i] = cut_txt
                            finished[i] = True

            # finalize
            for i in range(len(seqs)):
                if final_text[i] is None:
                    txt = self.tok_decode(gen_tokens[i])
                    final_text[i] = _truncate_at_any(txt, untils[i])
                outs.append(final_text[i])

            pbar.update(len(chunk))
        pbar.close()
        return outs


def _truncate_at_any(text: str, stops: List[str]) -> str:
    if not stops:
        return text
    cut = None
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    return text if cut is None else text[:cut]


# run_eval.py
if __name__ == "__main__":
    import csv
    import argparse
    from pathlib import Path
    from lm_eval.evaluator import simple_evaluate

    parser = argparse.ArgumentParser(description="Evaluate ChronoGPT on lm-eval benchmarks")
    parser.add_argument("--repo-id", type=str, default="manelalab/chrono-gpt-v1-20241231",
                        help="HuggingFace repo ID for the ChronoGPT model")
    CSR_TASKS = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    parser.add_argument("--tasks", type=str, nargs="+", default=CSR_TASKS,
                        help="Tasks to evaluate (default: Common Sense Reasoning suite)")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    model_name = args.repo_id
    lm = ChronoGPTLM(
        repo_id=model_name,
        cache_dir="cache",
        device="cuda",
        dtype="float16",
        batch_size=args.batch_size,
        max_gen_toks=512,
    )

    results = simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=lm.batch_size,
        limit=args.limit,
    )

    # --- Save to CSV ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name.replace('/', '_')}.csv"

    rows = []
    for task_name, task_results in results["results"].items():
        for metric, value in sorted(task_results.items()):
            if metric.endswith(",none") or metric.endswith(",flexible-extract"):
                rows.append({"task": task_name, "metric": metric, "score": f"{value:.4f}"})

    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "metric", "score"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n✅ Results saved to {out_file}")
    for r in rows:
        print(f"  {r['task']:30s} {r['metric']:25s} {r['score']}")
    print(results["results"])