"""tufazip v2: Transformer, packed-weight reader and online arithmetic codec.

V2_CONFIG holds the fixed settings for the verified enwik9 bitstream.
weights.bin supplies the Cycle2 initialization; encoding and decoding then
reproduce the same online learning steps. Deterministic decoding requires the
matching two-B200 hardware and pinned runtime documented in README.md.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")  # deterministic custom kernels

import argparse
import hashlib
import io
import json
import math
import time
import zipfile
import sys
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from arithmetic_coder import (ArithmeticEncoder, ArithmeticDecoder,
                              BitInputStream, BitOutputStream)
from checkpoint_utils import (
    CHECKPOINT_VERSION, REQUEUE_EXIT_CODE, atomic_torch_save,
    capture_rng_state, checkpoint_path, config_signature, flush_and_sync,
    load_checkpoint, mark_done, restore_rng_state, stop_requested)

# The fixed settings used to create the v2 enwik9 bitstream.
# The decoder must reproduce the same initialization and online updates.
V2_CONFIG = {'distributed': True,
 'raw_bytes': 1000000000,
 'vocab_bytes': 34421,
 'vocab_size': 4097,
 'tokenizer': 'word',
 'n_layer': 12,
 'd_model': 1024,
 'n_head': 8,
 'd_inner': 2816,
 'ctx_len': 192,
 'untie_embeddings': True,
 'dropout': 0.25,
 'batch_size': 64,
 'block_len': 500000,
 'online_loss': 'code',
 'project_only_targets': True,
 'lr': 4e-05,
 'min_lr_ratio': 0.25,
 'warmup': 200,
 'clip': 0.25,
 'adam_eps': 1e-09,
 'retrain_period': 1,
 'retrain_len': 35000000,
 'retrain_epochs': 1,
 'retrain_batch_size': 32,
 'retrain_update_len': 64,
 'retrain_lr': '1.6e-4,13000,1.6e-4,93000,7.9e-5,163000,4.0e-5,1911300,1.3e-5',
 'position_type': 'rope',
 'norm_style': 'peri',
 'dtype': 'bf16',
 'log_interval': 100000,
 'checkpoint_every_blocks': 4,
 'model_backend': 'transformer',
 'seed': 1111,
 'beta1': 0.0,
 'beta2': 0.9999}


LOG2 = math.log(2.0)
FREQ_TOTAL = 10_000_000  # frequency-table resolution handed to the arithmetic coder

# Files that must ship with this script to decompress its output, used for the
# Matt Mahoney "total size" benchmark (compressed bytes + DEFLATE-9 zip of these).
# Which detokenizer source is needed depends on the tokenizer: the C 'word' path
# needs the standalone preprocess.c source; the
# byte*/bpe* paths need tokenizers.py instead.
DECOMPRESSOR_COMMON = (
    "nn.py", "arithmetic_coder.py", "checkpoint_utils.py")
DECOMPRESSOR_MAMBA = (
    "ssm_models.py", "mamba_ssm_commit.txt", "mamba_ssm_compat.py")
DECOMPRESSOR_KDA = ("ssm_models.py", "kda_models.py", "fla_commit.txt")
DECOMPRESSOR_WORD = ("preprocess.c",)
DECOMPRESSOR_STRUCTURED = ("structured_word_transform.py",)
DECOMPRESSOR_PY = ("tokenizers.py",)


def decompressor_files(tokenizer, model_backend="transformer"):
    if tokenizer.startswith("word"):
        extra = DECOMPRESSOR_WORD
        if tokenizer != "word":
            extra += DECOMPRESSOR_STRUCTURED
    else:
        extra = DECOMPRESSOR_PY
    if model_backend in ("mamba2", "mamba3"):
        model_files = DECOMPRESSOR_MAMBA
    elif model_backend == "kda":
        model_files = DECOMPRESSOR_KDA
    else:
        model_files = ()
    return DECOMPRESSOR_COMMON + model_files + extra


def decompressor_zip_size(files, extra_files=()):
    root = os.path.dirname(os.path.abspath(__file__))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files:
            zf.write(os.path.join(root, f), arcname=f)
        for path in extra_files:
            zf.write(path, arcname=os.path.basename(path))
    return buf.tell()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        return normed.type_as(x) * self.weight


def build_rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos(), emb.sin()  # each [seq_len, head_dim]


def apply_rope(x, cos, sin):
    # x: [batch, n_head, seq, head_dim]; cos/sin: [seq, head_dim]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, d_model, n_head, dropout, ctx_len,
                 position_type="rope", qk_norm=True):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_model = d_model
        self.dropout = dropout
        self.position_type = position_type
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # QK-Norm (OLMo 2 / Gemma 3): per-head RMSNorm on q,k before RoPE.
        # Bounds attention logits, removes the spike pathology that needs
        # logit soft-capping, and lets us train at higher LR.
        self.q_norm = RMSNorm(self.d_head) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_head) if qk_norm else nn.Identity()
        if position_type == "learned_relative":
            # Bellard/Transformer-XL-style learned relative attention:
            # content bias, a content-dependent distance term, and a global
            # learned bias for every relative distance.
            self.r_emb = nn.Parameter(
                torch.empty(ctx_len, n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.empty(n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.zeros(ctx_len, n_head))
            nn.init.normal_(self.r_emb, mean=0.0, std=0.02)
            nn.init.normal_(self.r_w_bias, mean=0.0, std=0.02)

    def forward(self, x, cos=None, sin=None):
        batch, seq, d_model = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=2)
        q = q.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.position_type == "rope":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            # Direct relative-distance indexing is equivalent to the
            # Transformer-XL relative shift and is easier to audit here.
            pos = torch.arange(seq, device=x.device)
            distance = (pos[:, None] - pos[None, :]).clamp(min=0)
            r_bias = self.r_bias[distance]
            content_score = torch.einsum(
                "bhid,bhjd->bhij",
                q + self.r_w_bias[None, :, None, :], k)
            raw_position_score = torch.einsum(
                "bhid,khd->bhik", q, self.r_emb[:seq])
            position_score = raw_position_score.gather(
                -1, distance[None, None].expand(
                    batch, self.n_head, -1, -1))
            position_score = position_score + (
                r_bias.permute(2, 0, 1)[None] * math.sqrt(self.d_model))
            score = (content_score + position_score) / math.sqrt(self.d_head)
            causal_mask = torch.triu(
                torch.ones(seq, seq, dtype=torch.bool, device=x.device),
                diagonal=1)
            score = score.masked_fill(causal_mask[None, None], -float("inf"))
            prob = F.softmax(score.float(), dim=-1).to(v.dtype)
            prob = F.dropout(prob, p=self.dropout, training=self.training)
            out = torch.matmul(prob, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.gate = nn.Linear(d_model, d_inner, bias=False)
        self.up = nn.Linear(d_model, d_inner, bias=False)
        self.down = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_inner, dropout, ctx_len,
                 position_type="rope", qk_norm=True, norm_style="peri"):
        super().__init__()
        self.norm_style = norm_style
        self.attn_pre_norm = (
            RMSNorm(d_model) if norm_style == "peri" else nn.Identity())
        self.attn = Attention(
            d_model, n_head, dropout, ctx_len, position_type, qk_norm)
        self.attn_post_norm = RMSNorm(d_model)
        self.ffn_pre_norm = (
            RMSNorm(d_model) if norm_style == "peri" else nn.Identity())
        self.ffn = SwiGLU(d_model, d_inner)
        self.ffn_post_norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        if self.norm_style == "peri":
            x = x + self.drop(
                self.attn_post_norm(
                    self.attn(self.attn_pre_norm(x), cos, sin)))
            x = x + self.drop(
                self.ffn_post_norm(self.ffn(self.ffn_pre_norm(x))))
        else:
            x = self.attn_post_norm(
                x + self.drop(self.attn(x, cos, sin)))
            x = self.ffn_post_norm(
                x + self.drop(self.ffn(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_layer, d_model, n_head, d_inner,
                 ctx_len, dropout, rope_base=10000.0, tie_emb=True,
                 position_type="rope", qk_norm=True, norm_style="peri",
                 block_schedule=None):
        super().__init__()
        self.position_type = position_type
        if block_schedule is None:
            block_schedule = tuple(range(n_layer))
        else:
            block_schedule = tuple(int(index) for index in block_schedule)
            if not block_schedule:
                raise ValueError("block_schedule cannot be empty")
            if min(block_schedule) < 0 or max(block_schedule) >= n_layer:
                raise ValueError("block_schedule index is outside n_layer")
            if set(block_schedule) != set(range(n_layer)):
                raise ValueError(
                    "block_schedule must use every physical block")
        self.block_schedule = block_schedule
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Peri-LN also normalizes at the embedding boundary so the first block
        # sees the same well-conditioned input every other block sees.
        self.emb_norm = (
            RMSNorm(d_model) if norm_style == "peri" else nn.Identity())
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_inner, dropout, ctx_len,
                   position_type, qk_norm, norm_style)
             for _ in range(n_layer)])
        self.norm = (
            RMSNorm(d_model) if norm_style == "peri" else nn.Identity())
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_emb:
            self.head.weight = self.tok_emb.weight  # tie input and output embeddings

        if position_type == "rope":
            cos, sin = build_rope_cache(
                ctx_len, d_model // n_head, rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = self.rope_sin = None

        self.apply(self._init_weights)
        # Scale by executed rather than stored depth. This gives a looped model
        # the same initialization scale as the untied target it will become.
        residual_std = 0.02 / math.sqrt(2 * len(self.block_schedule))
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, output_positions=None):
        seq = idx.size(1)
        if self.position_type == "rope":
            cos, sin = self.rope_cos[:seq], self.rope_sin[:seq]
        else:
            cos = sin = None
        x = self.drop(self.emb_norm(self.tok_emb(idx)))
        for block_index in self.block_schedule:
            x = self.blocks[block_index](x, cos, sin)
        x = self.norm(x)
        # Online coding supervises only the newest position, and replay may
        # prepend context whose logits are never used.  Slice hidden states
        # before the large vocabulary projection when the caller identifies
        # the supervised positions.  The transformer and its context remain
        # unchanged; only provably-unused output rows are omitted.
        if output_positions is not None:
            x = x[:, output_positions]
        return self.head(x)


# Packed offline weights: QPI1 header, tensor metadata, scales and bit codes.
ARCHIVE_VERSION = 1


def _packed_size(num_values, bits):
    if bits == 16:
        return num_values * 2
    return (num_values * bits + 7) // 8

def _unpack_codes(payload, count, bits):
    packed = np.frombuffer(payload, dtype=np.uint8)
    if bits == 8:
        return packed[:count].copy()
    per_byte = 8 // bits
    shifts = np.arange(per_byte, dtype=np.uint16) * bits
    values = (
        (packed.astype(np.uint16)[:, None] >> shifts[None, :])
        & ((1 << bits) - 1))
    return values.reshape(-1)[:count].astype(np.uint8)

def read_tensor_archive(path):
    """Decode an archive into the exact float32 tensors seen by inference."""
    with open(path, "rb") as source:
        magic = source.read(4)
        if magic not in (b"QPR1", b"QPI1", b"LRA1", b"QLA1"):
            raise ValueError("Unknown tensor archive magic: {!r}".format(magic))
        length_bytes = source.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Truncated tensor archive header")
        metadata_length = struct.unpack("<Q", length_bytes)[0]
        metadata_payload = source.read(metadata_length)
        if len(metadata_payload) != metadata_length:
            raise ValueError("Truncated tensor archive metadata")
        metadata = json.loads(metadata_payload.decode("utf-8"))
        if metadata.get("version") != ARCHIVE_VERSION:
            raise ValueError("Unsupported tensor archive version")

        tensors = {}
        for entry in metadata["entries"]:
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Archive tensor name must be a non-empty string")
            if name in tensors:
                raise ValueError("Duplicate archive tensor: {}".format(name))
            shape = tuple(entry["shape"])
            if not shape or any(
                    not isinstance(size, int) or size < 1 for size in shape):
                raise ValueError("Invalid tensor shape for {}".format(name))
            count = int(np.prod(shape))
            bits = int(entry["bits"])
            if bits not in (1, 2, 4, 8, 16):
                raise ValueError("Unsupported bit width for {}".format(name))
            expected_value_bytes = _packed_size(count, bits)
            if int(entry["value_bytes"]) != expected_value_bytes:
                raise ValueError("Invalid packed size for {}".format(name))
            expected_scale_count = (
                shape[0] if len(shape) >= 2 else 1) if bits < 16 else 0
            if int(entry["scale_count"]) != expected_scale_count:
                raise ValueError("Invalid scale count for {}".format(name))
            if int(entry["scale_bytes"]) != expected_scale_count * 2:
                raise ValueError("Invalid scale size for {}".format(name))
            if bits == 16:
                payload = source.read(entry["value_bytes"])
                if len(payload) != entry["value_bytes"]:
                    raise ValueError(
                        "Truncated values for {}".format(entry["name"]))
                value = np.frombuffer(
                    payload, dtype=np.float16).astype(
                        np.float32).reshape(shape)
            else:
                scale_payload = source.read(entry["scale_bytes"])
                if len(scale_payload) != entry["scale_bytes"]:
                    raise ValueError(
                        "Truncated scales for {}".format(entry["name"]))
                scales = np.frombuffer(
                    scale_payload, dtype=np.float16).astype(np.float32)
                value_payload = source.read(entry["value_bytes"])
                if len(value_payload) != entry["value_bytes"]:
                    raise ValueError(
                        "Truncated values for {}".format(entry["name"]))
                codes = _unpack_codes(value_payload, count, bits)
                if bits == 1:
                    signed = codes.astype(np.int16) * 2 - 1
                else:
                    signed = (
                        codes.astype(np.int16)
                        - ((1 << (bits - 1)) - 1))
                if len(shape) >= 2:
                    scale_shape = (shape[0],) + (1,) * (len(shape) - 1)
                else:
                    scale_shape = (1,)
                value = (
                    signed.astype(np.float32).reshape(shape)
                    * scales.reshape(scale_shape))
            tensors[name] = torch.from_numpy(
                np.array(value, dtype=np.float32, copy=True))

        if source.read(1):
            raise ValueError("Tensor archive has trailing bytes")
    metadata["magic"] = magic.decode("ascii")
    metadata["archive_bytes"] = os.path.getsize(path)
    return tensors, metadata


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def depth_block_mapping(source_depth, target_depth):
    """Evenly place ordered source blocks in a deeper target model."""
    source_depth = int(source_depth)
    target_depth = int(target_depth)
    if source_depth < 1 or source_depth > target_depth:
        raise ValueError(
            "source depth must be between 1 and target depth")
    return [index * target_depth // source_depth
            for index in range(source_depth)]


def parse_block_schedule(value, n_layer):
    """Parse a comma-separated recurrent-depth schedule."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            schedule = [int(item) for item in value.split(",")]
        except ValueError as error:
            raise ValueError(
                "block_schedule must contain comma-separated integers") from error
    else:
        schedule = [int(item) for item in value]
    if not schedule:
        raise ValueError("block_schedule cannot be empty")
    if min(schedule) < 0 or max(schedule) >= int(n_layer):
        raise ValueError("block_schedule index is outside n_layer")
    if set(schedule) != set(range(int(n_layer))):
        raise ValueError("block_schedule must use every physical block")
    return schedule


def transformer_geometry(
        vocab_size, n_layer, d_model, n_head, d_inner, ctx_len,
        tie_embeddings, position_type, qk_norm, norm_style):
    return {
        "vocab_size": int(vocab_size),
        "n_layer": int(n_layer),
        "d_model": int(d_model),
        "n_head": int(n_head),
        "d_inner": int(d_inner),
        "ctx_len": int(ctx_len),
        "tie_embeddings": bool(tie_embeddings),
        "position_type": str(position_type),
        "qk_norm": bool(qk_norm),
        "norm_style": str(norm_style),
    }


def model_state_sha256(model):
    """Hash names, shapes, dtypes, and exact CPU parameter bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_depth_prior(model, archive_path, vocab_file, expected_geometry, seed):
    """Load a QPI1 depth-growth or loop-untying initialization."""
    if not vocab_file:
        raise ValueError("--vocab_file is required with --initial_prior")
    if not os.path.isfile(vocab_file):
        raise ValueError("Vocabulary file does not exist: {}".format(vocab_file))
    tensors, metadata = read_tensor_archive(archive_path)
    if metadata.get("magic") != "QPI1":
        raise ValueError("--initial_prior requires a QPI1 archive")
    initialization = metadata.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("QPI1 archive lacks initialization metadata")
    initialization_kind = initialization.get("kind")
    if initialization_kind not in ("depth-growth", "loop-untying"):
        raise ValueError("Unsupported initialization kind")
    if initialization.get("target_geometry") != expected_geometry:
        raise ValueError("Prior target geometry differs from this invocation")
    source_geometry = initialization.get("source_geometry")
    if not isinstance(source_geometry, dict):
        raise ValueError("QPI1 archive lacks source geometry")
    for key, value in expected_geometry.items():
        if key != "n_layer" and source_geometry.get(key) != value:
            raise ValueError(
                "Prior source geometry differs at {}".format(key))
    source_depth = int(source_geometry.get("n_layer", 0))
    target_depth = int(expected_geometry["n_layer"])
    if initialization_kind == "depth-growth":
        mapping = depth_block_mapping(source_depth, target_depth)
        if initialization.get("block_mapping") != mapping:
            raise ValueError(
                "Prior block mapping is not the canonical depth mapping")
        target_sources = {
            target_index: source_index
            for source_index, target_index in enumerate(mapping)
        }
        block_schedule = None
    else:
        block_schedule = parse_block_schedule(
            initialization.get("block_schedule"), source_depth)
        if block_schedule is None:
            raise ValueError("Loop prior lacks a block schedule")
        if len(block_schedule) != target_depth:
            raise ValueError(
                "Loop prior schedule length differs from target depth")
        mapping = list(block_schedule)
        target_sources = {
            target_index: source_index
            for target_index, source_index in enumerate(block_schedule)
        }
    if int(initialization.get("initialization_seed", -1)) != int(seed):
        raise ValueError("Prior initialization seed differs from --seed")
    vocab_digest = sha256_file(vocab_file)
    if initialization.get("vocab_sha256") != vocab_digest:
        raise ValueError("Prior vocabulary digest differs from --vocab_file")

    target_state = model.state_dict()
    root_names = {
        name for name in target_state if not name.startswith("blocks.")}
    block_suffixes = {
        name.split(".", 2)[2]
        for name in target_state if name.startswith("blocks.0.")}
    expected_source_names = set(root_names)
    for source_index in range(source_depth):
        expected_source_names.update(
            "blocks.{}.{}".format(source_index, suffix)
            for suffix in block_suffixes)
    if set(tensors) != expected_source_names:
        missing = sorted(expected_source_names - set(tensors))
        extra = sorted(set(tensors) - expected_source_names)
        raise ValueError(
            "Prior tensor set is incomplete: missing={} extra={}".format(
                missing, extra))

    with torch.no_grad():
        for name in root_names:
            if tuple(tensors[name].shape) != tuple(target_state[name].shape):
                raise ValueError("Prior tensor shape differs for {}".format(name))
            target_state[name].copy_(tensors[name].to(target_state[name].dtype))
        for target_index, source_index in sorted(target_sources.items()):
            for suffix in block_suffixes:
                source_name = "blocks.{}.{}".format(source_index, suffix)
                target_name = "blocks.{}.{}".format(target_index, suffix)
                if tuple(tensors[source_name].shape) != tuple(
                        target_state[target_name].shape):
                    raise ValueError(
                        "Prior tensor shape differs for {}".format(source_name))
                target_state[target_name].copy_(
                    tensors[source_name].to(target_state[target_name].dtype))
        if initialization_kind == "depth-growth":
            mapped_targets = set(target_sources)
            for target_index in range(target_depth):
                if target_index in mapped_targets:
                    continue
                target_state[
                    "blocks.{}.attn.proj.weight".format(target_index)].zero_()
                target_state[
                    "blocks.{}.ffn.down.weight".format(target_index)].zero_()

    return {
        "archive_bytes": os.path.getsize(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "vocab_sha256": vocab_digest,
        "initialization_kind": initialization_kind,
        "source_depth": source_depth,
        "block_mapping": mapping,
        "block_schedule": block_schedule,
        "initial_state_sha256": model_state_sha256(model),
        "metadata": metadata,
    }


def validate_checkpoint_prior(resume_state, prior_info):
    """Require resume to use the exact transmitted initialization identity."""
    requested_prior_digest = (
        prior_info["archive_sha256"] if prior_info is not None else None)
    if resume_state.get("initial_prior_sha256") != requested_prior_digest:
        raise ValueError(
            "Checkpoint initial-prior digest differs from this invocation")
    requested_vocab_digest = (
        prior_info["vocab_sha256"] if prior_info is not None else None)
    if resume_state.get(
            "initial_prior_vocab_sha256") != requested_vocab_digest:
        raise ValueError(
            "Checkpoint prior-vocabulary digest differs from this invocation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_tokens(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=">u2").astype(np.int64)


def lr_at(step, peak_lr, min_lr, warmup, total_steps):
    """Linear warmup followed by cosine decay to min_lr."""
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    if step >= total_steps:
        return min_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def parse_piecewise_lr(value):
    """Parse lr0[,step1,lr1,...] into a linear piecewise schedule."""
    parts = str(value).split(",")
    if len(parts) % 2 == 0:
        raise ValueError(
            "LR schedule must have syntax lr0[,step1,lr1,...], got {!r}".format(value))
    schedule = [(0, float(parts[0]))]
    previous_step = 0
    for index in range(1, len(parts), 2):
        step = int(parts[index])
        lr = float(parts[index + 1])
        if step <= previous_step:
            raise ValueError("LR schedule steps must be strictly increasing")
        schedule.append((step, lr))
        previous_step = step
    return schedule


def piecewise_lr_at(step, schedule):
    """Linearly interpolate a parsed schedule at integer optimizer step."""
    if step <= schedule[0][0]:
        return schedule[0][1]
    for (start_step, start_lr), (end_step, end_lr) in zip(
            schedule, schedule[1:]):
        if step <= end_step:
            fraction = (step - start_step) / (end_step - start_step)
            return start_lr + fraction * (end_lr - start_lr)
    return schedule[-1][1]


def make_freq_table(logits):
    """Cumulative integer frequency table per row, as the arithmetic coder wants."""
    prob = F.softmax(logits.detach(), dim=-1)
    freq = torch.clamp(torch.round(prob * FREQ_TOTAL).to(torch.int32), min=1)
    return torch.cumsum(freq, dim=-1).cpu().numpy()


class NullSummaryWriter:
    """Rank-local no-op writer; only rank zero owns durable logs."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_hparams(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="tufazip v2 neural codec")
    # I/O
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, help="output file (defaults under --work_dir)")
    p.add_argument("--decompress", action="store_true")
    p.add_argument("--no_coding", action="store_true",
                   help="skip the arithmetic coder; just train and report bits/token")
    p.add_argument("--max_tokens", type=int, default=0,
                   help="limit number of input tokens (0 = all), for quick tests")
    p.add_argument("--raw_bytes", type=int, default=0,
                   help="size of the original pre-tokenization file in bytes; "
                        "enables bits/byte logging (the enwik benchmark metric)")
    p.add_argument("--vocab_bytes", type=int, default=0,
                   help="serialized vocabulary size for separate benchmark-total metrics")
    p.add_argument("--vocab_file", type=str, default="",
                   help="serialized vocabulary whose SHA-256 is pinned by a QPI1 prior")
    p.add_argument("--initial_prior", type=str, default="",
                   help="transmitted QPI1 shallow-model initialization archive")
    p.add_argument("--external_decompressor_bytes", type=int, default=-1,
                   help="compressed bytes for external runtime dependencies; required "
                        "before reporting a non-Transformer benchmark total "
                        "(-1 = unaccounted)")
    # tokenizer: does not change modeling, only selects which decompressor source
    # files count toward total size, and labels the run. Set by the run script.
    p.add_argument("--tokenizer", type=str, default="word")
    # model
    p.add_argument("--model_backend",
                   choices=["transformer", "mamba2", "mamba3", "kda"],
                   default="transformer")
    p.add_argument("--vocab_size", type=int, default=16388)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_head", type=int, default=8)
    # d_inner default follows the Llama 2 SwiGLU convention:
    #   ceil(8/3 * d_model / 256) * 256
    # The 8/3 ratio makes the gated FFN's parameter count match a 4x ReLU FFN
    # (SwiGLU has 3 matrices vs ReLU's 2, so 4 * 2/3 = 8/3).  For d_model=512:
    #   8/3 * 512 = 1365.33  ->  rounded up to multiple of 256  ->  1536.
    p.add_argument("--d_inner", type=int, default=1536)
    # ctx_len = 512 is a conservative bump from the original 128: the Python
    # arithmetic coder dominates wall time, so the ~4x extra compute per step
    # is mostly absorbed, and longer windows capture entity/structure references
    # that span far more than 128 tokens.  Bellard's NNCP v2 effectively used
    # ~1024; go higher if your GPU has headroom.
    p.add_argument("--ctx_len", type=int, default=512, help="context window length")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--untie_embeddings", action="store_true",
                   help="use a separate output embedding instead of tying it to the input")
    p.add_argument("--position_type", choices=["rope", "learned_relative"],
                   default="rope",
                   help="RoPE or Bellard-style learned relative attention")
    p.add_argument("--no_qk_norm", action="store_true",
                   help="disable per-head RMSNorm on queries and keys")
    p.add_argument("--norm_style", choices=["peri", "post"], default="peri",
                   help="Peri-RMSNorm or Post-RMSNorm residual blocks")
    p.add_argument("--ssm_d_state", type=int, default=128)
    p.add_argument("--ssm_expand", type=int, default=2)
    p.add_argument("--ssm_headdim", type=int, default=64)
    p.add_argument("--ssm_d_conv", type=int, default=4,
                   help="Mamba-2 local convolution width")
    p.add_argument("--ssm_chunk_size", type=int, default=0,
                   help="kernel chunk size; 0 selects the upstream recommendation")
    p.add_argument("--mamba3_mimo_rank", type=int, default=4,
                   help="Mamba-3 MIMO rank; 1 selects the SISO kernel")
    p.add_argument("--kda_head_dim", type=int, default=128)
    p.add_argument("--kda_expand_v", type=float, default=1.0)
    p.add_argument("--kda_conv_size", type=int, default=4)
    p.add_argument("--kda_no_short_conv", action="store_true")
    p.add_argument("--kda_allow_neg_eigval", action="store_true")
    p.add_argument("--kda_safe_gate", action="store_true")
    p.add_argument("--kda_lower_bound", type=float, default=None)
    # training / online learning
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_len", type=int, default=500000,
                   help="tokens per block; streams are batched within a block")
    # 5e-4 vs the older 3e-4: QK-Norm caps attention logits and Peri-LN bounds
    # residual variance growth, so the model tolerates a modestly higher peak LR.
    p.add_argument("--lr", type=float, default=5e-4, help="peak learning rate")
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--online_loss", choices=["window", "code"], default="window",
                   help="window repeats all context targets (legacy); code trains only "
                        "the newly coded target, matching NNCP's online exposure")
    p.add_argument("--project_only_targets", action="store_true",
                   help="apply the vocabulary head only to supervised positions; "
                        "requires --online_loss code and changes no transformer "
                        "context or model parameters")
    p.add_argument("--clip", type=float, default=0.25)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    # retrain: extra passes over the already-coded prefix (valid because both the
    # compressor and decompressor hold it). On by default; --retrain_period 0 disables.
    p.add_argument("--retrain_period", type=int, default=500000,
                   help="retrain on the coded prefix every N symbols, checked at "
                        "block boundaries so the effective period rounds up to a "
                        "multiple of --block_len (0 = never)")
    p.add_argument("--retrain_len", type=int, default=10_000_000,
                   help="train on at most this many trailing symbols of the prefix")
    p.add_argument("--retrain_epochs", type=int, default=1,
                   help="passes over the prefix window per retrain")
    p.add_argument("--retrain_early_epochs", type=int, default=0,
                   help="override --retrain_epochs for replay calls at or before "
                        "--retrain_early_until symbols (0 disables)")
    p.add_argument("--retrain_early_until", type=int, default=0,
                   help="coded-prefix position through which to use "
                        "--retrain_early_epochs (0 disables)")
    p.add_argument("--retrain_lr", type=str, default="3e-4",
                   help="retrain LR schedule: lr0[,step1,lr1,...], with linear "
                        "interpolation and a global step counter")
    p.add_argument("--retrain_update_len", type=int, default=0,
                   help="new retraining targets per optimizer update; 0 uses ctx_len")
    p.add_argument("--retrain_batch_size", type=int, default=0,
                   help="independent retraining stream count; 0 uses batch_size")
    p.add_argument("--seed", type=int, default=1111)
    # runtime
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--distributed", action="store_true",
                   help="run a deterministic single-node DDP codec under torchrun; "
                        "world size is part of the codec and must match on decode/resume")
    p.add_argument("--work_dir", type=str, default="runs")
    p.add_argument("--log_interval", type=int, default=100000,
                   help="report interval in symbols")
    p.add_argument("--run_name", type=str, default="",
                   help="suffix appended to the run dir, shown as the TensorBoard run name")
    p.add_argument("--run_dir", type=str, default="",
                   help="exact TensorBoard/log directory (stable across Slurm requeues)")
    p.add_argument("--checkpoint_dir", type=str, default="",
                   help="persistent directory for latest.pt; an existing checkpoint "
                        "is resumed automatically")
    p.add_argument("--checkpoint_every_blocks", type=int, default=2,
                   help="periodic checkpoint interval in completed blocks; a Slurm "
                        "stop request always checkpoints at the next block boundary")
    p.add_argument("--stop_after_blocks", type=int, default=0,
                   help=argparse.SUPPRESS)
    p.set_defaults(**V2_CONFIG,
                   vocab_file=os.path.join(os.path.dirname(__file__), "enwik9.voc"),
                   initial_prior=os.path.join(os.path.dirname(__file__), "weights.bin"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.checkpoint_every_blocks < 1:
        raise ValueError("--checkpoint_every_blocks must be at least 1")
    if args.project_only_targets and args.online_loss != "code":
        raise ValueError(
            "--project_only_targets requires --online_loss code")
    if args.retrain_epochs < 1:
        raise ValueError("--retrain_epochs must be at least 1")
    if (args.retrain_early_epochs == 0) != (args.retrain_early_until == 0):
        raise ValueError(
            "--retrain_early_epochs and --retrain_early_until must either both "
            "be set or both be 0")
    if args.retrain_early_epochs < 0 or args.retrain_early_until < 0:
        raise ValueError("early replay settings cannot be negative")
    if args.external_decompressor_bytes < -1:
        raise ValueError("--external_decompressor_bytes must be -1 or nonnegative")
    if args.initial_prior and args.model_backend != "transformer":
        raise ValueError("--initial_prior currently supports only Transformer models")
    if args.initial_prior and not os.path.isfile(args.initial_prior):
        raise ValueError("Initial prior does not exist: {}".format(
            args.initial_prior))

    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size > 1 and not args.distributed:
        raise ValueError("torchrun world size > 1 requires --distributed")
    if args.distributed:
        if args.cpu or not torch.cuda.is_available():
            raise ValueError("--distributed currently requires CUDA")
        if launched_world_size < 2:
            raise ValueError("--distributed requires torchrun with at least two ranks")
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(
            "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    is_main = rank == 0
    amp_enabled = device.type == "cuda" and args.dtype == "bf16"
    if args.model_backend != "transformer" and device.type != "cuda":
        raise ValueError("Mamba and KDA backends require a CUDA GPU")
    if args.distributed and args.model_backend != "transformer":
        raise ValueError(
            "The deterministic DDP pilot currently supports only the Transformer backend")

    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # run dir name: timestamp + key configs (dataset from --input, model family from
    # the script name) so runs are distinguishable at a glance; full config is in HParams.
    model_tag = "modern" if args.model_backend == "transformer" else args.model_backend
    dataset_tag = os.path.splitext(os.path.basename(args.input))[0]
    run_tag = "{}-{}-{}-{}-v{}-L{}-d{}-ctx{}".format(
        time.strftime("%Y%m%d-%H%M%S"), model_tag, dataset_tag,
        args.tokenizer, args.vocab_size,
        args.n_layer, args.d_model, args.ctx_len)
    if args.run_name:
        run_tag += "-" + args.run_name
    run_dir = args.run_dir or os.path.join(args.work_dir, run_tag)
    if args.distributed:
        run_dir_holder = [run_dir if is_main else None]
        dist.broadcast_object_list(run_dir_holder, src=0)
        run_dir = run_dir_holder[0]
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    if args.distributed:
        dist.barrier()
    if args.output is None:
        args.output = os.path.join(run_dir, "out.bin")
    log_path = os.path.join(run_dir, "log.txt")
    writer = SummaryWriter(log_dir=run_dir) if is_main else NullSummaryWriter()

    def log(msg):
        if not is_main:
            return
        print(msg, flush=True)
        os.makedirs(run_dir, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    retrain_lr_schedule = parse_piecewise_lr(args.retrain_lr)
    if args.model_backend == "transformer":
        model = Transformer(
            args.vocab_size, args.n_layer, args.d_model, args.n_head,
            args.d_inner, args.ctx_len, args.dropout,
            tie_emb=not args.untie_embeddings,
            position_type=args.position_type,
            qk_norm=not args.no_qk_norm,
            norm_style=args.norm_style)
    elif args.model_backend in ("mamba2", "mamba3"):
        from ssm_models import MambaLanguageModel
        model = MambaLanguageModel(
            args.vocab_size, args.n_layer, args.d_model, args.dropout,
            tie_emb=not args.untie_embeddings,
            model_backend=args.model_backend,
            d_state=args.ssm_d_state,
            expand=args.ssm_expand,
            headdim=args.ssm_headdim,
            d_conv=args.ssm_d_conv,
            chunk_size=args.ssm_chunk_size,
            mamba3_mimo_rank=args.mamba3_mimo_rank,
            amp_dtype=args.dtype)
    else:
        from kda_models import KDALanguageModel
        model = KDALanguageModel(
            args.vocab_size, args.n_layer, args.d_model, args.d_inner,
            args.dropout, tie_emb=not args.untie_embeddings,
            head_dim=args.kda_head_dim,
            expand_v=args.kda_expand_v,
            use_short_conv=not args.kda_no_short_conv,
            conv_size=args.kda_conv_size,
            allow_neg_eigval=args.kda_allow_neg_eigval,
            safe_gate=args.kda_safe_gate,
            lower_bound=args.kda_lower_bound)
    prior_info = None
    if args.initial_prior:
        expected_geometry = transformer_geometry(
            args.vocab_size, args.n_layer, args.d_model, args.n_head,
            args.d_inner, args.ctx_len, not args.untie_embeddings,
            args.position_type, not args.no_qk_norm, args.norm_style)
        prior_info = load_depth_prior(
            model, args.initial_prior, args.vocab_file,
            expected_geometry, args.seed)
    model_core = model.to(device)
    # Gradients work in eval mode. Keeping online coding in eval mode lets a
    # matched profile enable dropout only during prefix retraining, as NNCP does.
    model_core.eval()
    if args.distributed:
        model = DistributedDataParallel(
            model_core, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False)
        # Initialization must be identical before DDP synchronizes parameters;
        # subsequent rank-specific RNG streams make replay dropout independent
        # while remaining exactly reproducible on encode, decode, and resume.
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
    else:
        model = model_core
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.beta1, args.beta2), eps=args.adam_eps)
    # separate optimizer so retrain's Adam moments don't perturb the online-coding ones
    retrain_optimizer = torch.optim.Adam(
                                         model.parameters(),
                                         lr=retrain_lr_schedule[0][1],
                                         betas=(args.beta1, args.beta2), eps=args.adam_eps)

    n_params = sum(p.numel() for p in model_core.parameters())
    emb_params = model_core.tok_emb.weight.numel()
    if model_core.head.weight is not model_core.tok_emb.weight:
        emb_params += model_core.head.weight.numel()
    n_params_nonemb = n_params - emb_params
    log("=" * 72)
    for k, v in vars(args).items():
        log("    - {} : {}".format(k, v))
    log("    - device : {}".format(device))
    log("    - distributed_world_size : {}".format(world_size))
    log("=" * 72)
    log("#params = {} (#non-embedding = {})".format(n_params, n_params_nonemb))
    base_dfiles = decompressor_files(args.tokenizer, args.model_backend)
    base_decompressor_bytes = decompressor_zip_size(base_dfiles)
    if prior_info is not None:
        dfiles = base_dfiles
        local_decompressor_bytes = decompressor_zip_size(
            dfiles, extra_files=(args.initial_prior,))
    else:
        dfiles = base_dfiles
        local_decompressor_bytes = base_decompressor_bytes
    prior_transmission_bytes = (
        local_decompressor_bytes - base_decompressor_bytes)
    dependency_accounted = (
        args.model_backend == "transformer" or
        args.external_decompressor_bytes >= 0)
    decompressor_bytes = (
        local_decompressor_bytes + max(0, args.external_decompressor_bytes))
    log("local decompressor zip ({} files) = {} bytes".format(
        len(dfiles), local_decompressor_bytes))
    if prior_info is not None:
        log("initial prior raw={} bytes transmitted_zip_delta={} bytes "
            "sha256={} kind={} source_depth={} mapping={} "
            "state_sha256={}".format(
                prior_info["archive_bytes"], prior_transmission_bytes,
                prior_info["archive_sha256"],
                prior_info["initialization_kind"],
                prior_info["source_depth"],
                prior_info["block_mapping"],
                prior_info["initial_state_sha256"]))
    if args.model_backend in ("mamba2", "mamba3"):
        log("SSM kernel chunk size = {}".format(model_core.effective_chunk_size))
    elif args.model_backend == "kda":
        log("KDA kernel mode = {}".format(model_core.kernel_mode))
    if args.model_backend != "transformer":
        if dependency_accounted:
            log("external model runtime archive = {} bytes".format(
                args.external_decompressor_bytes))
        else:
            log("WARNING: model runtime is not packaged/charged; "
                "benchmark-total metrics are disabled for this experimental run")

    signature_excluded = {
        "input", "output", "work_dir", "run_name", "run_dir",
        "checkpoint_dir", "checkpoint_every_blocks", "stop_after_blocks",
        # Paths may move between encoder and decoder. QPI1 contents are pinned
        # separately by the archive and vocabulary SHA-256 values.
        "initial_prior", "vocab_file",
        # Accounting does not affect predictions or optimizer updates.
        "external_decompressor_bytes",
        # Stored and checked separately below so checkpoints written
        # before this option existed remain resumable in legacy mode.
        "project_only_targets",
        # Likewise, old checkpoints predate the optional early replay
        # override. Missing values mean the disabled/default policy.
        "retrain_early_epochs", "retrain_early_until",
    }
    mamba_signature_args = {
        "ssm_d_state", "ssm_expand", "ssm_headdim", "ssm_d_conv",
        "ssm_chunk_size", "mamba3_mimo_rank",
    }
    kda_signature_args = {
        "kda_head_dim", "kda_expand_v", "kda_conv_size",
        "kda_no_short_conv", "kda_allow_neg_eigval", "kda_safe_gate",
        "kda_lower_bound",
    }
    if args.model_backend == "transformer":
        # Keep pre-SSM Transformer checkpoint signatures byte-for-byte stable.
        signature_excluded.update(
            {"model_backend"} | mamba_signature_args | kda_signature_args)
    elif args.model_backend in ("mamba2", "mamba3"):
        # KDA options were added later and do not change Mamba predictions.
        signature_excluded.update(kda_signature_args)
    else:
        signature_excluded.update(mamba_signature_args)
    checkpoint_signature = config_signature(args, excluded=signature_excluded)
    resume_state = (
        load_checkpoint(args.checkpoint_dir, device)
        if args.checkpoint_dir else None)
    if resume_state is not None:
        if resume_state.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported checkpoint version")
        if resume_state.get("config_signature") != checkpoint_signature:
            raise ValueError(
                "Checkpoint configuration differs from this invocation")
        if bool(resume_state.get("project_only_targets", False)) != (
                args.project_only_targets):
            raise ValueError(
                "Checkpoint target-projection mode differs from this invocation")
        if (int(resume_state.get("retrain_early_epochs", 0)) !=
                args.retrain_early_epochs or
                int(resume_state.get("retrain_early_until", 0)) !=
                args.retrain_early_until):
            raise ValueError(
                "Checkpoint early replay policy differs from this invocation")
        if resume_state.get("input_size") != os.path.getsize(args.input):
            raise ValueError("Checkpoint input size differs from --input")
        checkpoint_world_size = int(resume_state.get("distributed_world_size", 1))
        if checkpoint_world_size != world_size:
            raise ValueError(
                "Checkpoint world size {} differs from this invocation {}"
                .format(checkpoint_world_size, world_size))
        validate_checkpoint_prior(resume_state, prior_info)
        model_core.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        retrain_optimizer.load_state_dict(resume_state["retrain_optimizer"])
        log("resuming checkpoint {} at block_start={} symbols={}".format(
            checkpoint_path(args.checkpoint_dir),
            resume_state["block_start"], resume_state["n_symbols"]))

    decode = args.decompress
    code = not args.no_coding
    batch = args.batch_size
    ctx_len = args.ctx_len

    # --- set up input/output and the arithmetic coder ---
    out_file = in_file = encoder = decoder = bit_out = bit_in = None
    if not decode:
        file_data = load_tokens(args.input).copy()
        if args.max_tokens:
            file_data = file_data[:args.max_tokens]
        original_len = len(file_data)
        if code and is_main:
            if resume_state is not None:
                out_file = open(args.output, "r+b")
            else:
                out_file = open(args.output, "wb")
                out_file.write(original_len.to_bytes(4, byteorder="big"))
            bit_out = BitOutputStream(out_file)
            encoder = ArithmeticEncoder(32, bit_out)
            if resume_state is not None:
                bit_out.load_state_dict(resume_state["bitstream"])
                encoder.load_state_dict(resume_state["arithmetic_coder"])
    else:
        assert code, "--no_coding cannot be combined with --decompress"
        if is_main:
            in_file = open(args.input, "rb")
            original_len = int.from_bytes(in_file.read(4), byteorder="big")
            bit_in = BitInputStream(in_file)
            decoder = ArithmeticDecoder(32, bit_in)
            if resume_state is not None:
                bit_in.load_state_dict(resume_state["bitstream"])
                decoder.load_state_dict(resume_state["arithmetic_coder"])
                out_file = open(args.output, "r+b")
                output_position = int(resume_state["output_position"])
                out_file.seek(0)
                decoded_prefix = out_file.read(output_position)
                out_file.seek(output_position)
                out_file.truncate(output_position)
            else:
                out_file = open(args.output, "wb")
                decoded_prefix = b""
        if args.distributed:
            original_len_holder = [original_len if is_main else None]
            dist.broadcast_object_list(original_len_holder, src=0)
            original_len = int(original_len_holder[0])
            # Each rank needs the already-decoded prefix to reconstruct future
            # contexts after a resume. Rank zero owns the mutable file handle;
            # peers perform a bounded read-only load from the durable output.
            if resume_state is not None and not is_main:
                output_position = int(resume_state["output_position"])
                with open(args.output, "rb") as decoded_file:
                    decoded_prefix = decoded_file.read(output_position)
            elif not is_main:
                decoded_prefix = b""
        else:
            # The non-distributed path initialized these values on rank zero.
            pass
        file_data = np.zeros(original_len, dtype=np.int64)
        if decoded_prefix:
            prefix_tokens = np.frombuffer(
                decoded_prefix, dtype=">u2").astype(np.int64)
            file_data[:len(prefix_tokens)] = prefix_tokens

    if resume_state is not None and resume_state["original_len"] != original_len:
        raise ValueError("Checkpoint token count differs from --input")

    # pad the token array up to a whole number of batch streams
    file_len = ((original_len + batch - 1) // batch) * batch
    if len(file_data) < file_len:
        file_data = np.concatenate(
            [file_data, np.zeros(file_len - len(file_data), dtype=np.int64)])
    block_len = max(batch, (args.block_len // batch) * batch)
    total_steps = file_len // batch
    min_lr = args.lr * args.min_lr_ratio
    log("tokens = {}, steps = {}".format(original_len, total_steps))
    # bits/byte = bits/token / bytes_per_token; the model only sees tokens, so this
    # uses the whole-file average token length (None when --raw_bytes is unset).
    bytes_per_token = args.raw_bytes / original_len if args.raw_bytes else None

    def source_progress(symbol_step):
        """Map an encoded-token position onto the corresponding source bytes."""
        if not args.raw_bytes:
            return int(symbol_step)
        bounded_step = min(int(symbol_step), original_len)
        return bounded_step * args.raw_bytes // original_len

    def add_scalar(tag, value, symbol_step):
        """Log legacy token-step and cross-tokenizer source-step views."""
        writer.add_scalar(tag, value, symbol_step)
        if args.raw_bytes:
            writer.add_scalar(
                "by_source/{}".format(tag), value,
                source_progress(symbol_step))

    # --- main online compression / decompression loop ---
    if batch % world_size:
        raise ValueError(
            "batch_size {} must be divisible by distributed world size {}"
            .format(batch, world_size))
    arange_batch = np.arange(batch)
    local_batch = batch // world_size
    local_batch_start = rank * local_batch
    local_batch_end = local_batch_start + local_batch
    retrain_batch = args.retrain_batch_size or batch
    if retrain_batch <= 0:
        raise ValueError("retrain_batch_size must be positive")
    if retrain_batch % world_size:
        raise ValueError(
            "retrain_batch_size {} must be divisible by distributed world size {}"
            .format(retrain_batch, world_size))
    retrain_arange_batch = np.arange(retrain_batch)
    local_retrain_batch = retrain_batch // world_size
    local_retrain_start = rank * local_retrain_batch
    local_retrain_end = local_retrain_start + local_retrain_batch
    vocab_size = args.vocab_size

    retrain_train_step = (
        int(resume_state["retrain_train_step"]) if resume_state else 0)

    def retrain(file_data, file_pos):
        """Extra training passes over the already-coded prefix file_data[:file_pos].

        Reads only data both sides already hold at this point, so the weights it
        produces are reproducible during decompression and the codec stays valid.
        Runs on a separate optimizer, leaving the online-coding Adam state untouched."""
        nonlocal retrain_train_step
        lo = max(0, file_pos - args.retrain_len)
        stride = (file_pos - lo) // retrain_batch
        if stride < 2:
            return
        global_stream_base = lo + retrain_arange_batch * stride
        stream_base = global_stream_base[local_retrain_start:local_retrain_end]
        n_targets = stride - 1
        model.train()
        window_bits = 0.0
        current_retrain_lr = retrain_lr_schedule[0][1]
        update_len = args.retrain_update_len or ctx_len
        if update_len > ctx_len:
            raise ValueError("retrain_update_len must be <= ctx_len")
        retrain_epochs = (
            args.retrain_early_epochs
            if (args.retrain_early_until
                and file_pos <= args.retrain_early_until)
            else args.retrain_epochs)
        for epoch in range(retrain_epochs):
            epoch_nats = 0.0
            for start in range(0, n_targets, update_len):
                length = min(update_len, n_targets - start)
                # Each target is optimized exactly once, while earlier known
                # tokens provide up to ctx_len positions of conditioning.
                # Keep the zero/default path byte-for-byte equivalent to the
                # original non-overlapping retraining implementation.
                context_start = (max(0, start - (ctx_len - length))
                                 if args.retrain_update_len else start)
                prefix_len = start - context_start
                cols = context_start + np.arange(prefix_len + length)
                inp = file_data[stream_base[:, None] + cols[None, :]]
                tgt_cols = start + np.arange(length)
                tgt = file_data[stream_base[:, None] + tgt_cols[None, :] + 1]
                inp_t = torch.from_numpy(inp).to(device)
                tgt_t = torch.from_numpy(tgt).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp_enabled):
                    if args.project_only_targets:
                        target_logits = model(
                            inp_t,
                            output_positions=slice(
                                prefix_len, prefix_len + length))
                    else:
                        logits = model(inp_t)
                        target_logits = logits[
                            :, prefix_len:prefix_len + length]
                loss = F.cross_entropy(target_logits.float().reshape(-1, vocab_size),
                                       tgt_t.reshape(-1))
                retrain_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                current_retrain_lr = piecewise_lr_at(
                    retrain_train_step, retrain_lr_schedule)
                for group in retrain_optimizer.param_groups:
                    group["lr"] = current_retrain_lr
                retrain_optimizer.step()
                retrain_train_step += 1
                metric_loss = loss.detach().float()
                if args.distributed:
                    dist.all_reduce(metric_loss, op=dist.ReduceOp.SUM)
                    metric_loss /= world_size
                epoch_nats += metric_loss.item() * length
            window_bits = epoch_nats / n_targets / LOG2
        model.eval()
        log("R {:8d}  window={:8d}  epochs={:2d}  bits/tok={:6.3f}  "
            "step={:7d}  lr={:.3g}".format(
                file_pos, file_pos - lo, retrain_epochs, window_bits,
                retrain_train_step, current_retrain_lr))
        add_scalar("retrain/window_bits", window_bits, file_pos)
        add_scalar("retrain/window_perplexity", math.exp(window_bits * LOG2), file_pos)
        add_scalar("retrain/epochs", retrain_epochs, file_pos)
        add_scalar("retrain/lr", current_retrain_lr, file_pos)
        add_scalar("retrain/step", retrain_train_step, file_pos)

    now = time.time()
    if resume_state is not None:
        n_symbols = int(resume_state["n_symbols"])
        n_bits = float(resume_state["n_bits"])
        train_step = int(resume_state["train_step"])
        block_start = int(resume_state["block_start"])
        last_retrain_pos = int(resume_state["last_retrain_pos"])
        start_time = now - float(resume_state["elapsed_seconds"])
        # Do not include queue time in the first post-resume throughput sample.
        last_symbols, last_bits, last_time = n_symbols, n_bits, now
        if args.distributed:
            rng_by_rank = resume_state.get("rng_by_rank")
            if rng_by_rank is None or len(rng_by_rank) != world_size:
                raise ValueError("Distributed checkpoint is missing per-rank RNG state")
            restore_rng_state(rng_by_rank[rank])
        else:
            restore_rng_state(resume_state["rng"])
    else:
        n_symbols = last_symbols = 0
        n_bits = last_bits = 0.0
        start_time = last_time = now
        train_step = 0
        block_start = 0
        last_retrain_pos = 0

    completed_blocks = block_start // block_len

    def save_resume_checkpoint(reason):
        if not args.checkpoint_dir:
            return
        local_rng = capture_rng_state()
        rng_by_rank = None
        if args.distributed:
            rng_by_rank = [None] * world_size if is_main else None
            dist.gather_object(local_rng, rng_by_rank, dst=0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if args.distributed and not is_main:
            dist.barrier()
            return
        flush_and_sync(out_file)
        coder_state = stream_state = None
        output_position = None
        if encoder is not None:
            coder_state = encoder.state_dict()
            stream_state = bit_out.state_dict()
        elif decoder is not None:
            coder_state = decoder.state_dict()
            stream_state = bit_in.state_dict()
            output_position = out_file.tell()
        state = {
            "version": CHECKPOINT_VERSION,
            "config_signature": checkpoint_signature,
            "project_only_targets": args.project_only_targets,
            "retrain_early_epochs": args.retrain_early_epochs,
            "retrain_early_until": args.retrain_early_until,
            "input_size": os.path.getsize(args.input),
            "original_len": original_len,
            "file_len": file_len,
            "distributed_world_size": world_size,
            "initial_prior_sha256": (
                prior_info["archive_sha256"] if prior_info is not None else None),
            "initial_prior_vocab_sha256": (
                prior_info["vocab_sha256"] if prior_info is not None else None),
            "model": model_core.state_dict(),
            "optimizer": optimizer.state_dict(),
            "retrain_optimizer": retrain_optimizer.state_dict(),
            "rng": local_rng if not args.distributed else None,
            "rng_by_rank": rng_by_rank,
            "arithmetic_coder": coder_state,
            "bitstream": stream_state,
            "output_position": output_position,
            "n_symbols": n_symbols,
            "n_bits": n_bits,
            "train_step": train_step,
            "retrain_train_step": retrain_train_step,
            "block_start": block_start,
            "last_retrain_pos": last_retrain_pos,
            "elapsed_seconds": time.time() - start_time,
            "reason": reason,
        }
        checkpoint_started = time.time()
        atomic_torch_save(state, checkpoint_path(args.checkpoint_dir))
        writer.flush()
        log("checkpoint saved at block_start={} ({}, {:.1f}s)".format(
            block_start, reason, time.time() - checkpoint_started))
        if args.distributed:
            dist.barrier()
    # column header for the periodic progress rows (bpt = bits/token, bpb = bits/byte;
    # loc = this interval, cum = cumulative; bpb columns are nan unless --raw_bytes is set)
    log("{:>8} {:>11} {:>11} {:>7} {:>7} {:>8} {:>8} {:>7} {:>9}".format(
        "step", "symbols", "comp_bytes", "bpt_loc", "bpt_cum",
        "bpb_loc", "bpb_cum", "kS/s", "lr"))
    while block_start < file_len:
        if args.retrain_period and block_start - last_retrain_pos >= args.retrain_period:
            retrain(file_data, block_start)
            last_retrain_pos = block_start
        cur_block_len = min(file_len - block_start, block_len)
        block_stride = cur_block_len // batch
        global_stream_base = block_start + arange_batch * block_stride  # [batch]
        stream_base = global_stream_base[local_batch_start:local_batch_end]

        for stream_pos in range(block_stride):
            window = min(stream_pos, ctx_len)
            if window == 0:
                inp = np.zeros(
                    (local_batch, 1), dtype=np.int64)  # dummy BOS for the first token
            else:
                offsets = stream_pos - window + np.arange(window)
                inp = file_data[stream_base[:, None] + offsets[None, :]]
            inp_t = torch.from_numpy(inp).to(device)

            lr = lr_at(train_step, args.lr, min_lr, args.warmup, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp_enabled):
                if args.project_only_targets:
                    logits = model(
                        inp_t, output_positions=slice(-1, None))
                else:
                    logits = model(inp_t)           # [batch, seq, vocab]
            code_logits = logits[:, -1, :].float()  # next-token logits used for coding

            # arithmetic-code (or decode) one symbol per stream at this position
            global_code_pos = global_stream_base + stream_pos
            code_pos = stream_base + stream_pos
            if code:
                if args.distributed:
                    gathered_logits = (
                        [torch.empty_like(code_logits) for _ in range(world_size)]
                        if is_main else None)
                    dist.gather(code_logits.detach(), gathered_logits, dst=0)
                    if is_main:
                        global_code_logits = torch.cat(gathered_logits, dim=0)
                else:
                    global_code_logits = code_logits
                if is_main:
                    freq = make_freq_table(global_code_logits)
                    for i in range(batch):
                        if decode:
                            file_data[global_code_pos[i]] = decoder.read(freq[i])
                        else:
                            encoder.write(
                                freq[i], int(file_data[global_code_pos[i]]))

            if decode and args.distributed:
                global_code_targets = torch.empty(
                    batch, dtype=torch.long, device=device)
                if is_main:
                    global_code_targets.copy_(torch.from_numpy(
                        file_data[global_code_pos]).to(device))
                dist.broadcast(global_code_targets, src=0)
                decoded_targets = global_code_targets.cpu().numpy()
                file_data[global_code_pos] = decoded_targets
                code_targets = global_code_targets[
                    local_batch_start:local_batch_end]
            else:
                code_targets = torch.from_numpy(file_data[code_pos]).to(device)

            # bits accounting uses only the coded symbols (= compressed size)
            code_nats = F.cross_entropy(
                code_logits, code_targets, reduction="sum").detach()
            if args.distributed:
                dist.all_reduce(code_nats, op=dist.ReduceOp.SUM)
            n_bits += code_nats.item() / LOG2
            n_symbols += batch

            # The legacy modern experiment trained every overlapping target in
            # the window. `code` trains only the new symbol, like NNCP v2, so
            # each online target has approximately equal exposure.
            if args.online_loss == "code":
                loss = F.cross_entropy(code_logits, code_targets)
            else:
                if window == 0:
                    targets = file_data[code_pos][:, None]
                else:
                    target_offsets = stream_pos - window + 1 + np.arange(window)
                    targets = file_data[stream_base[:, None] + target_offsets[None, :]]
                targets_t = torch.from_numpy(targets).to(device)
                loss = F.cross_entropy(logits.float().reshape(-1, vocab_size),
                                       targets_t.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_step += 1

            is_last = (block_start + cur_block_len >= file_len
                       and stream_pos == block_stride - 1)
            if n_symbols - last_symbols >= args.log_interval or is_last:
                now = time.time()
                local_bpt = (n_bits - last_bits) / (n_symbols - last_symbols)
                cum_bpt = n_bits / n_symbols
                local_bpb = local_bpt / bytes_per_token if bytes_per_token else float("nan")
                cum_bpb = cum_bpt / bytes_per_token if bytes_per_token else float("nan")
                metric_loss = loss.detach().float()
                if args.distributed:
                    dist.all_reduce(metric_loss, op=dist.ReduceOp.SUM)
                    metric_loss /= world_size
                loss_nats = metric_loss.item()
                ks_per_s = (n_symbols - last_symbols) / ((now - last_time) * 1000.0)
                weight_norm = (
                    torch.sqrt(sum(p.detach().float().pow(2).sum()
                                   for p in model.parameters())).item()
                    if is_main else 0.0)
                log("{:8d} {:11d} {:11.0f} {:7.3f} {:7.3f} {:8.4f} {:8.4f} {:7.2f} {:9.3g}".format(
                    train_step, n_symbols, n_bits / 8.0, local_bpt, cum_bpt,
                    local_bpb, cum_bpb, ks_per_s, lr))
                add_scalar("bits_per_token/local", local_bpt, n_symbols)
                add_scalar("bits_per_token/cumulative", cum_bpt, n_symbols)
                if bytes_per_token:
                    add_scalar("bits_per_byte/local", local_bpb, n_symbols)
                    add_scalar("bits_per_byte/cumulative", cum_bpb, n_symbols)
                # the real compression signal: perplexity of the next-token coding loss
                add_scalar("code/perplexity_local", math.exp(local_bpt * LOG2), n_symbols)
                add_scalar("code/perplexity_cumulative", math.exp(cum_bpt * LOG2), n_symbols)
                # window-averaged training loss (dominated by already-seen tokens; not compression)
                add_scalar("train/window_bits", loss_nats / LOG2, n_symbols)
                add_scalar("train/window_perplexity", math.exp(loss_nats), n_symbols)
                add_scalar("optim/lr", lr, n_symbols)
                add_scalar("optim/grad_norm", grad_norm.item(), n_symbols)
                add_scalar("optim/weight_norm", weight_norm, n_symbols)
                add_scalar("perf/kSymbols_per_s", ks_per_s, n_symbols)
                add_scalar("size/bytes", n_bits / 8.0, n_symbols)
                if dependency_accounted:
                    add_scalar("size/total", n_bits / 8.0 + args.vocab_bytes +
                               decompressor_bytes, n_symbols)
                if device.type == "cuda":
                    add_scalar("perf/gpu_mem_gb",
                               torch.cuda.max_memory_allocated() / 1e9, n_symbols)
                last_symbols, last_bits, last_time = n_symbols, n_bits, now

        if decode and is_main:
            end = min(block_start + cur_block_len, original_len)
            out_file.write(file_data[block_start:end].astype(">u2").tobytes())
            out_file.flush()
        block_start += cur_block_len
        completed_blocks += 1

        requested_stop = (
            args.checkpoint_dir and stop_requested(args.checkpoint_dir))
        test_stop = (
            args.stop_after_blocks
            and completed_blocks >= args.stop_after_blocks)
        periodic = (
            args.checkpoint_dir
            and completed_blocks % args.checkpoint_every_blocks == 0)
        if block_start < file_len and (periodic or requested_stop or test_stop):
            reason = (
                "stop-request" if requested_stop else
                "test-stop" if test_stop else "periodic")
            save_resume_checkpoint(reason)
        if block_start < file_len and (requested_stop or test_stop):
            log("stopping cleanly for Slurm requeue")
            writer.close()
            flush_and_sync(out_file)
            if out_file is not None:
                out_file.close()
            if in_file is not None:
                in_file.close()
            if args.distributed:
                dist.barrier()
                dist.destroy_process_group()
            return REQUEUE_EXIT_CODE

    if encoder is not None:
        encoder.finish()
        bit_out.close()
    if out_file is not None:
        out_file.close()
    if in_file is not None:
        in_file.close()
    if args.distributed:
        dist.barrier()

    total_time = time.time() - start_time
    compressed_path = args.input if decode else args.output
    actual_compressed_bytes = os.path.getsize(compressed_path) if code else None
    actual_bits_per_byte = (actual_compressed_bytes * 8 / args.raw_bytes
                            if actual_compressed_bytes is not None and args.raw_bytes
                            else None)
    benchmark_total_bytes = (
        actual_compressed_bytes + args.vocab_bytes + decompressor_bytes
        if actual_compressed_bytes is not None and dependency_accounted
        else None)
    benchmark_total_bpb = (benchmark_total_bytes * 8 / args.raw_bytes
                           if benchmark_total_bytes is not None and args.raw_bytes
                           else None)
    final_bpt = n_bits / n_symbols
    bits_per_byte = final_bpt / bytes_per_token if bytes_per_token else None
    # record the run config alongside its headline results in the HParams tab;
    # run_name="." keeps them in this run instead of a timestamped sub-run.
    metrics = {"hparam/bits_per_token": final_bpt,
               "hparam/perplexity": math.exp(final_bpt * LOG2),
               "hparam/bytes": n_bits / 8.0,
               "hparam/base_decompressor_bytes": base_decompressor_bytes,
               "hparam/local_decompressor_bytes": local_decompressor_bytes,
               "hparam/prior_transmission_bytes": prior_transmission_bytes,
               "hparam/vocab_bytes": args.vocab_bytes,
               "hparam/kSymbols_per_s": n_symbols / (total_time * 1000.0)}
    if prior_info is not None:
        metrics["hparam/prior_archive_bytes"] = prior_info["archive_bytes"]
        add_scalar("final/prior_transmission_bytes",
                   prior_transmission_bytes, n_symbols)
    if dependency_accounted:
        metrics["hparam/decompressor_bytes"] = decompressor_bytes
        metrics["hparam/total_size"] = (
            n_bits / 8.0 + args.vocab_bytes + decompressor_bytes)
    if actual_compressed_bytes is not None:
        metrics["hparam/actual_compressed_bytes"] = actual_compressed_bytes
        add_scalar("final/actual_compressed_bytes", actual_compressed_bytes,
                   n_symbols)
    if actual_bits_per_byte is not None:
        metrics["hparam/actual_bits_per_byte"] = actual_bits_per_byte
        add_scalar("final/actual_bits_per_byte", actual_bits_per_byte,
                   n_symbols)
    if benchmark_total_bytes is not None:
        metrics["hparam/benchmark_total_bytes"] = benchmark_total_bytes
        add_scalar("final/benchmark_total_bytes", benchmark_total_bytes,
                   n_symbols)
    if benchmark_total_bpb is not None:
        metrics["hparam/benchmark_total_bits_per_byte"] = benchmark_total_bpb
        add_scalar("final/benchmark_total_bits_per_byte",
                   benchmark_total_bpb, n_symbols)
    if bits_per_byte is not None:
        metrics["hparam/bits_per_byte"] = bits_per_byte
    writer.add_hparams(
        {**vars(args), "device": str(device),
         "n_params": n_params, "n_params_nonemb": n_params_nonemb},
        metrics, run_name=".",
    )
    writer.close()

    if benchmark_total_bytes is not None:
        log("benchmark total = {} payload + {} vocab + {} decompressor = {} "
            "bytes{}".format(
                actual_compressed_bytes, args.vocab_bytes, decompressor_bytes,
                benchmark_total_bytes,
                " ({:.6f} bits/byte)".format(benchmark_total_bpb)
                if benchmark_total_bpb is not None else ""))

    estimated_total_bytes = (
        n_bits / 8.0 + args.vocab_bytes + decompressor_bytes
        if dependency_accounted else None)
    final_result = {
        "symbols": n_symbols,
        "modeled_bits": n_bits,
        "modeled_payload_bytes": n_bits / 8.0,
        "modeled_bits_per_token": final_bpt,
        "modeled_bits_per_byte": bits_per_byte,
        "actual_compressed_bytes": actual_compressed_bytes,
        "actual_bits_per_byte": actual_bits_per_byte,
        "vocab_bytes": args.vocab_bytes,
        "base_decompressor_zip_bytes": base_decompressor_bytes,
        "decompressor_with_prior_zip_bytes": local_decompressor_bytes,
        "prior_archive_bytes": (
            prior_info["archive_bytes"] if prior_info is not None else 0),
        "prior_transmission_bytes": prior_transmission_bytes,
        "initial_prior_sha256": (
            prior_info["archive_sha256"] if prior_info is not None else None),
        "prior_source_depth": (
            prior_info["source_depth"] if prior_info is not None else 0),
        "prior_initialization_kind": (
            prior_info["initialization_kind"]
            if prior_info is not None else None),
        "prior_block_mapping": (
            prior_info["block_mapping"] if prior_info is not None else None),
        "prior_policy": (
            prior_info["metadata"].get("policy")
            if prior_info is not None else None),
        "estimated_benchmark_total_bytes": estimated_total_bytes,
        "estimated_benchmark_total_bits_per_byte": (
            estimated_total_bytes * 8.0 / args.raw_bytes
            if estimated_total_bytes is not None and args.raw_bytes else None),
        "benchmark_total_bytes": benchmark_total_bytes,
        "benchmark_total_bits_per_byte": benchmark_total_bpb,
        "elapsed_seconds": total_time,
    }
    log("FINAL_RESULT " + json.dumps(final_result, sort_keys=True))

    if dependency_accounted:
        size_summary = "{:.0f} bytes (+ {} decompressor = {:.0f} total)".format(
            n_bits / 8.0, decompressor_bytes, n_bits / 8.0 + decompressor_bytes)
    else:
        size_summary = (
            "{:.0f} payload bytes (model runtime not yet charged)"
            .format(n_bits / 8.0))
    log("done in {:.1f} s ({:.2f} kS/s), {:.4f} bits/token{}, {}".format(
        total_time, n_symbols / (total_time * 1000.0), final_bpt,
        ", {:.4f} bits/byte".format(bits_per_byte) if bits_per_byte is not None else "",
        size_summary))
    if args.checkpoint_dir and is_main:
        mark_done(args.checkpoint_dir)
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
