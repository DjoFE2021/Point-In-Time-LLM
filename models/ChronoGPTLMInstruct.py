"""
ChronoGPT Instruct model wrapper for AlpacaEval.

Loads the instruct-tuned ChronoGPT from HuggingFace Hub and provides
a simple generate() interface compatible with alpaca.py.

Usage in alpaca.py:
    from models.ChronoGPTLMInstruct import ChronoGPTInstruct

    wrapper = ChronoGPTInstruct.from_hub("manelalab/chrono-gpt-instruct-v1-20201231")
    responses = wrapper.generate_batch(prompts, max_new_tokens=512)
"""

import os
import json
import math
import gc
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import tiktoken


# ------------------------------------------------------------------
# Architecture components (matches HF instruct repo exactly)
# ------------------------------------------------------------------

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        if not torch.tensor(0).is_meta:
            self._create_buffers()
        else:
            self.register_buffer('cos', torch.empty(max_seq_len, dim, dtype=torch.float32), persistent=False)
            self.register_buffer('sin', torch.empty(max_seq_len, dim, dtype=torch.float32), persistent=False)

    def _create_buffers(self, device=None):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.dim // 4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        if device is not None:
            angular_freq = angular_freq.to(device)
            t = t.to(device)
        theta = torch.einsum('i,j -> ij', t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x):
        if self.cos.is_meta:
            self._create_buffers(device=x.device)
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

    def forward(self, x, ve):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        if self.kv_cache is not None:
            k = torch.cat([self.kv_cache[0], k], dim=1)
            v = torch.cat([self.kv_cache[1], v], dim=1)
            self.kv_cache = torch.stack([k, v])
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

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
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs):
        base = [emb(inputs).bfloat16() for emb in self.embed]
        L = self.num_layers
        half = L // 2
        encoder = [base[i] if i < 3 else None for i in range(half)]
        decoder = [base[i - (half - 3)] if i >= (half - 3) else None for i in range(half)]
        return encoder + decoder


# ------------------------------------------------------------------
# Core model (matches HF instruct repo -- forward returns logits only)
# ------------------------------------------------------------------

class ChronoGPT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, use_attn=True) for _ in range(num_layers)])
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers=num_layers)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

    @torch.inference_mode()
    def forward(self, inputs, past_key_values=None):
        B = inputs.size(0)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        x0 = norm(self.embed(inputs).bfloat16())
        x = x0

        ve = [self.value_embeds(inputs[i].view(-1)) for i in range(B)]
        ve = [torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None
              for i in range(len(ve[0]))]
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        if past_key_values is not None:
            for i, block in enumerate(self.blocks):
                if block.attn is not None:
                    block.attn.kv_cache = past_key_values[i]

        present = []
        skip_connections = []

        for i in range(self.num_encoder_layers):
            block = self.blocks[i]
            x = block(x, ve_enc[i], x0)
            if block.attn is not None:
                present.append(block.attn.kv_cache)
                block.attn.kv_cache = None
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            block = self.blocks[self.num_encoder_layers + i]
            x = block(x, ve_dec[i], x0)
            if block.attn is not None:
                present.append(block.attn.kv_cache)
                block.attn.kv_cache = None

        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)
        return logits.float()

    @classmethod
    def from_pretrained(cls, repo_id, cache_dir=None, **kwargs):
        config_path = hf_hub_download(repo_id=repo_id, filename="config.pt", cache_dir=cache_dir)
        bin_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=cache_dir)
        config = torch.load(config_path, map_location="cpu")
        model = cls(**config)
        model.load_state_dict(torch.load(bin_path, map_location="cpu"))
        return model


# ------------------------------------------------------------------
# High-level instruct wrapper for alpaca.py
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are ChronoGPT, a large language model trained by ManelaLab at WashU.\n"
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request."
)


def _format_instruct_prompt(instruction: str) -> str:
    """Format a raw instruction into the ChronoGPT instruct prompt template."""
    return (
        f"\n\n### Instruction:\n{SYSTEM_PROMPT}\n{instruction}\n\n"
        f"### Input:\n### Response:\n"
    )


class ChronoGPTInstruct:
    """
    Convenience wrapper that loads the instruct-tuned ChronoGPT and
    exposes a simple generate_batch() interface for alpaca.py.

    Uses the same tiktoken GPT-2 tokenizer and autoregressive generation
    as the HuggingFace repo's extract_response / generate functions.
    """

    def __init__(self, model: ChronoGPT, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eos_id = 50256
        self.context_size = 1792

    @classmethod
    def from_hub(
        cls,
        repo_id: str = "manelalab/chrono-gpt-instruct-v1-20201231",
        cache_dir: str = "cache",
        device: str = "cuda",
        dtype: str = "float16",
    ) -> "ChronoGPTInstruct":
        """Load the instruct model from HuggingFace Hub."""
        print(f"Loading ChronoGPT Instruct from {repo_id} ...")
        model = ChronoGPT.from_pretrained(repo_id, cache_dir=cache_dir)

        if dtype in ("float16", "fp16", "half"):
            model = model.half()
        elif dtype in ("bfloat16", "bf16"):
            model = model.bfloat16()

        model = model.to(device).eval()
        gc.collect()
        torch.cuda.empty_cache() if device.startswith("cuda") else None

        tokenizer = tiktoken.get_encoding("gpt2")
        print(f"  ChronoGPT Instruct loaded on {device}")
        return cls(model, tokenizer, device)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = None,
    ) -> str:
        """Generate a response for a single prompt (already formatted or raw instruction)."""
        EOT = "<" + "|endoftext|" + ">"
        token_ids = self.tokenizer.encode(prompt, allowed_special={EOT})
        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size:]
            logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float('-inf')).to(logits.device),
                    logits,
                )

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if idx_next.item() == self.eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)

        # Decode only the generated portion
        generated_ids = idx[0, len(token_ids):].tolist()
        text = self.tokenizer.decode(generated_ids)
        # Strip any trailing "### Response:" artifacts
        text = text.replace("### Response:", "").strip()
        return text

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = None,
        start_idx: int = 0,
    ) -> List[str]:
        """Generate responses for a list of prompts (for alpaca.py integration)."""
        outputs = []
        total = start_idx + len(prompts)
        for i, prompt in enumerate(prompts):
            response = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            outputs.append(response)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [ChronoGPT-Instruct] Generated {start_idx + i + 1}/{total}")
        return outputs

    @staticmethod
    def format_prompt(instruction: str) -> str:
        """Format an instruction using the ChronoGPT instruct template."""
        return _format_instruct_prompt(instruction)
