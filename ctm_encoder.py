"""
ctm_encoder.py — a Continuous-Thought-Machine event-stream encoder for the
medical-JEPA. Drop-in replacement for TransformerEncoder in train_jepa*.py.

WHY THIS IS NOT THE ROADMAP STUB
--------------------------------
The roadmap sketched a `CTMEncoder` that was a GRUCell looped a few times with
the per-step projections averaged. That averaging discards the *synchronization
representation*, which is the entire point of a Continuous Thought Machine — so
that stub is just a recurrent aggregator wearing a CTM label. If you ran the
MLP/Transformer/CTM bake-off with it and CTM lost, the result would be
uninterpretable (you wouldn't have tested CTM). This module implements the three
actual CTM mechanisms (Darlow et al., "Continuous Thought Machines", 2025):

  1. INTERNAL TICKS decoupled from input length. The model "thinks" for T ticks
     regardless of how many clinical events the patient has.
  2. NEURON-LEVEL MODELS (NLMs): every neuron has its OWN small MLP that maps a
     short history of its pre-activations to its next post-activation. Timing of
     a neuron's activity carries information, not just its instantaneous value.
  3. SYNCHRONIZATION REPRESENTATION: the latent the model actually reads out
     (and queries attention with) is built from the temporal correlation between
     pairs of neurons across ticks — not the raw activations.

I/O CONTRACT (identical to TransformerEncoder, so it's a true drop-in)
----------------------------------------------------------------------
  __init__(n_symptoms, n_labs, n_vitals, n_drug_buckets, n_modalities,
           z_dim=128, ...)
  forward(batch) where batch is the dict made by make_seq_batches:
      modality, feat_id : long  (B, L)
      value, t_hours    : float (B, L)
      pad_mask          : bool  (B, L), True = padding
  returns z : (B, z_dim)
  attribute is_sequence = True

The event embedding (per-modality tables + modality + value + time) is copied
from TransformerEncoder on purpose, so any difference in the bake-off is
attributable to the temporal core, not to a different input representation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTMEventEncoder(nn.Module):
    def __init__(self, n_symptoms, n_labs, n_vitals, n_drug_buckets, n_modalities,
                 z_dim=128, d_model=128, n_neurons=128, memory_length=8,
                 ticks=8, nlm_hidden=16, n_heads=4,
                 n_sync_action=128, n_sync_out=256, p=0.1, seed=0):
        super().__init__()
        self.is_sequence = True
        self.d_model = d_model
        self.D = n_neurons
        self.M = memory_length
        self.T = ticks

        # ── event embedding (mirrors TransformerEncoder for a fair comparison) ──
        self.sym_emb = nn.Embedding(n_symptoms + 1, d_model)
        self.lab_emb = nn.Embedding(n_labs + 1, d_model)
        self.vit_emb = nn.Embedding(n_vitals + 1, d_model)
        self.drug_emb = nn.Embedding(n_drug_buckets + 1, d_model)
        self.mod_emb = nn.Embedding(n_modalities, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.time_proj = nn.Linear(1, d_model)
        # a learnable "sink" token so attention always has something unmasked to
        # read (handles patients with an empty event window without NaNs).
        self.sink = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.MOD_SYMPTOM, self.MOD_LAB, self.MOD_VITAL, self.MOD_DRUG = 1, 2, 3, 4

        # ── CTM core ──
        # synapse model U(z_prev, attended_input) -> pre-activations for all neurons
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.q_proj = nn.Linear(n_sync_action, d_model)        # query <- action synchronization
        self.attn_to_neuron = nn.Linear(d_model, self.D)
        self.synapse = nn.Sequential(
            nn.Linear(2 * self.D, 2 * self.D), nn.LayerNorm(2 * self.D), nn.GELU(),
            nn.Dropout(p), nn.Linear(2 * self.D, self.D),
        )
        self.pre_norm = nn.LayerNorm(self.D)

        # Neuron-Level Models: a private 1-hidden-layer MLP per neuron, mapping
        # that neuron's M-step pre-activation history -> its post-activation.
        # Vectorised with per-neuron weight tensors.
        g = torch.Generator().manual_seed(seed)
        self.nlm_w1 = nn.Parameter(torch.randn(self.D, self.M, nlm_hidden, generator=g) * (1.0 / self.M ** 0.5))
        self.nlm_b1 = nn.Parameter(torch.zeros(self.D, nlm_hidden))
        self.nlm_w2 = nn.Parameter(torch.randn(self.D, nlm_hidden, 1, generator=g) * (1.0 / nlm_hidden ** 0.5))
        self.nlm_b2 = nn.Parameter(torch.zeros(self.D, 1))

        # learnable starting post-activation and pre-activation trace
        self.z0 = nn.Parameter(torch.zeros(self.D))
        self.trace0 = nn.Parameter(torch.zeros(self.D))

        # ── synchronization: random neuron pairs (i,j); sync_ij = decayed
        #    temporal correlation of post-activations across ticks. Two subsets:
        #    one drives attention (action), one is the read-out latent (out).  ──
        gi = torch.Generator().manual_seed(seed + 1)
        def _pairs(n):
            i = torch.randint(0, self.D, (n,), generator=gi)
            j = torch.randint(0, self.D, (n,), generator=gi)
            return i, j
        ai, aj = _pairs(n_sync_action)
        oi, oj = _pairs(n_sync_out)
        self.register_buffer("act_i", ai); self.register_buffer("act_j", aj)
        self.register_buffer("out_i", oi); self.register_buffer("out_j", oj)
        # learnable per-pair decay rate (>=0 via softplus); 0 => uniform over ticks
        self.act_decay = nn.Parameter(torch.zeros(n_sync_action))
        self.out_decay = nn.Parameter(torch.zeros(n_sync_out))
        self.out = nn.Linear(n_sync_out, z_dim)

    # ── embedding identical in spirit to the transformer baseline ──
    def _embed(self, batch):
        modality = batch["modality"]; feat_id = batch["feat_id"]
        value = batch["value"].unsqueeze(-1); t_hours = batch["t_hours"].unsqueeze(-1)
        pad_mask = batch["pad_mask"]
        B, L = modality.shape
        fe = torch.zeros(B, L, self.d_model, device=modality.device)
        for mod, table in ((self.MOD_SYMPTOM, self.sym_emb), (self.MOD_LAB, self.lab_emb),
                           (self.MOD_VITAL, self.vit_emb), (self.MOD_DRUG, self.drug_emb)):
            m = (modality == mod)
            if m.any():
                ids = feat_id.clamp(min=0) * m
                fe = fe + table(ids) * m.unsqueeze(-1)
        tok = (fe + self.mod_emb(modality) + self.value_proj(value)
               + self.time_proj(torch.log1p(t_hours.clamp(min=0)) / 5.0))
        # prepend the always-visible sink token
        sink = self.sink.expand(B, -1, -1)
        kv = torch.cat([sink, tok], dim=1)                              # (B, L+1, d)
        sink_pad = torch.zeros(B, 1, dtype=torch.bool, device=modality.device)
        key_pad = torch.cat([sink_pad, pad_mask], dim=1)                # (B, L+1)
        return kv, key_pad

    def _nlm(self, history):
        # history: (B, D, M) -> post-activations (B, D)
        h = torch.einsum("bdm,dmh->bdh", history, self.nlm_w1) + self.nlm_b1
        h = F.gelu(h)
        o = torch.einsum("bdh,dho->bdo", h, self.nlm_w2) + self.nlm_b2
        return torch.tanh(o.squeeze(-1))

    def forward(self, batch):
        kv, key_pad = self._embed(batch)
        B = kv.size(0)
        dev = kv.device

        z = self.z0.unsqueeze(0).expand(B, -1).contiguous()             # (B, D)
        # pre-activation history buffer, last M ticks, seeded with trace0
        hist = self.trace0.view(1, self.D, 1).expand(B, self.D, self.M).contiguous()

        # running synchronization accumulators (numerator + normaliser)
        a_decay = torch.sigmoid(self.act_decay)                          # in (0,1)
        o_decay = torch.sigmoid(self.out_decay)
        a_num = torch.zeros(B, self.act_i.numel(), device=dev)
        a_den = torch.zeros_like(a_num)
        o_num = torch.zeros(B, self.out_i.numel(), device=dev)
        o_den = torch.zeros_like(o_num)

        def _update_sync(num, den, decay, idx_i, idx_j):
            prod = z[:, idx_i] * z[:, idx_j]                             # (B, S)
            num = decay * num + prod
            den = decay * den + 1.0
            return num, den, num / torch.sqrt(den + 1e-6)

        # seed synchronization from the initial post-activation
        a_num, a_den, a_sync = _update_sync(a_num, a_den, a_decay, self.act_i, self.act_j)
        o_num, o_den, o_sync = _update_sync(o_num, o_den, o_decay, self.out_i, self.out_j)

        for _ in range(self.T):
            # 1. query attention from the current action-synchronization
            q = self.q_proj(a_sync).unsqueeze(1)                         # (B, 1, d)
            attended, _ = self.attn(q, kv, kv, key_padding_mask=key_pad) # (B, 1, d)
            attended = attended.squeeze(1)                               # (B, d)
            # 2. synapse model: combine recurrent state + attended input -> pre-acts
            drive = self.attn_to_neuron(attended)                        # (B, D)
            pre = self.pre_norm(self.synapse(torch.cat([z, drive], dim=-1)))
            # 3. roll the pre-activation history and append
            hist = torch.cat([hist[:, :, 1:], pre.unsqueeze(-1)], dim=-1)
            # 4. neuron-level models -> new post-activations
            z = self._nlm(hist)
            # 5. update synchronization
            a_num, a_den, a_sync = _update_sync(a_num, a_den, a_decay, self.act_i, self.act_j)
            o_num, o_den, o_sync = _update_sync(o_num, o_den, o_decay, self.out_i, self.out_j)

        return self.out(o_sync)                                          # (B, z_dim)


if __name__ == "__main__":
    # smoke test against the exact batch contract used by make_seq_batches
    torch.manual_seed(0)
    B, L = 2, 5
    batch = {
        "modality": torch.tensor([[1, 2, 3, 4, 0], [1, 2, 0, 0, 0]]),
        "feat_id":  torch.tensor([[3, 5, 1, 9, 0], [2, 7, 0, 0, 0]]),
        "value":    torch.tensor([[0., 15., 93., 0., 0.], [0., 110., 0., 0., 0.]]),
        "t_hours":  torch.tensor([[0., 0., 0., 6., 0.], [0., 0., 0., 0., 0.]]),
        "pad_mask": torch.tensor([[False, False, False, False, True],
                                  [False, False, True, True, True]]),
    }
    enc = CTMEventEncoder(n_symptoms=179, n_labs=67, n_vitals=9,
                          n_drug_buckets=256, n_modalities=5, z_dim=64)
    z = enc(batch)
    n_params = sum(p.numel() for p in enc.parameters())
    print("output z:", tuple(z.shape), "finite:", bool(torch.isfinite(z).all()))
    print("params:", f"{n_params/1e3:.1f}K")
    # gradient check
    z.sum().backward()
    g = sum(p.grad.abs().sum().item() for p in enc.parameters() if p.grad is not None)
    print("grad flows:", g > 0)
