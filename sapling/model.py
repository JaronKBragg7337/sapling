"""Sapling core — a decoder-only transformer, trained from scratch.

v1/v2: ~0.82M parameters (d 128, 4 layers, 64 tokens of context) - word parts.
v3 (10/14, Jaron: "school level 2: real sentences"): ~12M parameters (d 384,
6 layers, 6 heads, 256 tokens of context) - the same class, bigger numbers, so
every checkpoint from gen 0 still loads. Fits the RTX 4060 (8 GB) with room;
trains the whole school in minutes there, and runs (slowly) on CPU.

The core is the *owned* seed; what it has met lives first in memory and is
consolidated into these weights by the heartbeat (v2, no gate).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

V3_CONFIG = dict(d_model=384, n_head=6, n_layer=6, block_size=256)
# v3.1 (10/14 night): the same core with twice the context (a document chunk + a question fit).
V31_CONFIG = dict(d_model=384, n_head=6, n_layer=6, block_size=512)
# v4 (10/15, Claude as teacher): a bigger core, trained fresh with the same tokenizer, rotary positions (DESIGN-CHOICES D24).
V4_CONFIG = dict(d_model=640, n_head=10, n_layer=10, block_size=512, rope=True)


def _rope_cache(T: int, hs: int, device, dtype, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hs, 2, device=device, dtype=torch.float32) / hs))
    f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    f = torch.cat((f, f), dim=-1)                            # rotate-half layout: pairs (i, i + hs/2)
    return f.cos().to(dtype)[None, None], f.sin().to(dtype)[None, None]   # (1, 1, T, hs)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    rot = torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    return x * cos + rot * sin


def autocast_ctx(device: str):
    """bf16 autocast on CUDA (the 4060 supports it; ~2x faster, same results in practice); no-op on CPU."""
    from contextlib import nullcontext
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int, rope: bool = False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.rope = rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        if self.rope:          # v4: rotary positions - relative offsets, the shape a copy circuit needs
            cos, sin = _rope_cache(T, hs, x.device, q.dtype)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        # v3: the fused kernel (same math as the explicit mask; ~2-3x faster at 256 tokens)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int, rope: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SaplingCore(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_head: int = 4,
                 n_layer: int = 4, block_size: int = 64, rope: bool = False):
        super().__init__()
        self.block_size = block_size
        self.rope = rope
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # v4: with rotary positions there is no learned position table (positions live inside attention)
        self.pos_emb = None if rope else nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, block_size, rope) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        # v3: small init (GPT-2 style). The default N(0,1) embedding with a tied head put the
        # first loss at ~43 (ln(vocab) is ~7); old checkpoints are unaffected (weights are loaded over this).
        self.apply(self._init_weights)
        # GPT-2: residual output projections scaled by 1/sqrt(2 * layers) (only matters for a fresh core)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layer) ** 0.5)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, eos_id: int,
                 temperature: float = 0.0, top_k: int = 40, seed: int | None = None,
                 repeat_penalty: float = 1.0) -> torch.Tensor:
        """Greedy decoding by default: deterministic, reproducible (fly-school
        rule: every trial reproducible from its seed). Exams and chat answers
        use greedy. temperature > 0 = seeded top-k sampling, used only for free
        writing (E8 / /write) and, from v4, the chat box - still reproducible from its seed.
        repeat_penalty > 1 (v4 chat only): logits of tokens it already wrote in this answer are divided by it
        (CTRL-style), so a small core does not loop on one phrase."""
        start = idx.shape[1]
        gen = None
        if temperature > 0:
            gen = torch.Generator(device=idx.device)
            gen.manual_seed(int(seed if seed is not None else 0))
        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()
            if repeat_penalty != 1.0 and idx.shape[1] > start:
                prev = idx[:, start:]
                vals = logits.gather(1, prev)
                logits.scatter_(1, prev, torch.where(vals > 0, vals / repeat_penalty, vals * repeat_penalty))
            if gen is None:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=gen)
            idx = torch.cat([idx, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return idx

    def config(self) -> dict:
        blk = self.blocks[0]
        cfg = dict(vocab_size=self.tok_emb.weight.shape[0], d_model=self.tok_emb.weight.shape[1],
                   n_head=blk.attn.n_head, n_layer=len(self.blocks), block_size=self.block_size)
        if self.rope:
            cfg["rope"] = True          # old checkpoints keep their exact config (no key = learned positions)
        return cfg

    @torch.no_grad()
    def grow_vocab(self, new_size: int) -> int:
        """Add rows to the (tied) token embedding so the vocabulary can grow
        as it meets new word parts (Jaron 10/14: "it should be able to grow
        its vocabulary as it learns"). New rows start at the mean of the
        existing rows plus small noise, so an unseen token behaves like a
        generic word until training moves it. Returns rows added."""
        old = self.tok_emb.weight
        if new_size <= old.shape[0]:
            return 0
        added = new_size - old.shape[0]
        new_emb = nn.Embedding(new_size, old.shape[1]).to(old.device)
        new_emb.weight[: old.shape[0]] = old
        mean = old.mean(dim=0, keepdim=True)
        # noise at the spread of the existing rows: new tokens must be TELLABLE APART
        # from each other from the start (at 0.02 they all collapsed into one word)
        new_emb.weight[old.shape[0]:] = mean + old.std(dim=0, keepdim=True) * torch.randn(added, old.shape[1], device=old.device)
        self.tok_emb = new_emb
        self.head = nn.Linear(old.shape[1], new_size, bias=False).to(old.device)
        self.head.weight = self.tok_emb.weight
        return added

    @torch.no_grad()
    def grow_context(self, new_block: int) -> int:
        """v3.1: lengthen the context in place (256 -> 512). The learned position
        rows it has stay; the new positions start as copies of the last learned
        row plus small noise, so a long prompt is readable at once and training
        moves them. Returns rows added. The same individual keeps its weights.
        v4 (rotary): there is no table; the window just gets longer."""
        if self.pos_emb is None:
            added = max(0, new_block - self.block_size)
            self.block_size = max(self.block_size, new_block)
            return added
        old = self.pos_emb.weight
        if new_block <= old.shape[0]:
            return 0
        added = new_block - old.shape[0]
        new_pos = nn.Embedding(new_block, old.shape[1]).to(old.device)
        new_pos.weight[: old.shape[0]] = old
        new_pos.weight[old.shape[0]:] = old[-1:] + 0.02 * torch.randn(added, old.shape[1], device=old.device)
        self.pos_emb = new_pos
        self.block_size = new_block
        return added

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
