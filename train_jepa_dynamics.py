"""
train_jepa.py — Phase 2: the actual medical-JEPA training scaffold.

⚠️  REQUIRES PyTorch + (ideally) a GPU. This does NOT run in the Osler
    container — it's written to run on your own machine / a GPU box.
    Everything UP TO this file (schema, converter, labeler, vocab,
    featurizer) is pure-Python and already verified. This is the piece
    that needs hardware.

What this implements (the smallest useful JEPA):
  - Context encoder:  patient observations  → z_context
  - Target encoder:   (EMA copy of context encoder) → z_target   [stop-grad]
  - Predictor:        z_context + time-delta → z_hat
  - Loss:             cosine distance(z_hat, sg(z_target))   [the JEPA loss]
  - Plus an auxiliary state-decoder head with a masked, confidence-weighted
    regression loss on the OBSERVABLE states (the supervised anchor).

This follows the I-JEPA / V-JEPA recipe: predict in representation space,
target encoder is an EMA of the context encoder, gradient only flows through
the context encoder + predictor. The auxiliary state head is what makes the
latent decodable / probeable (and gives the observable states real supervision).

Run:
    pip install torch numpy
    python3 build_vocab.py full          # produces vocab.json
    python3 train_jepa.py --data osler_coverage_trajectories.jsonl --epochs 50

For real training you'd point --data at a MIMIC-derived JSONL (same schema),
not the synthetic coverage set.
"""
from __future__ import annotations
from ctm_encoder import CTMEventEncoder
import argparse, os, sys, json, math, random

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── hard dependency check with a friendly message ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit(
        "\n[train_jepa] PyTorch not found.\n"
        "This script needs torch (and ideally a GPU). Install it where you have\n"
        "hardware:  pip install torch numpy\n"
        "Then:      python3 build_vocab.py full && python3 train_jepa.py\n"
    )

try:
    from featurize_dynamics import build_examples, load_vocab
except ImportError:
    from featurize import build_examples, load_vocab


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """MLP encoder: patient observation VECTOR → latent embedding z.
    Consumes the flat aggregated context from featurize.py."""
    def __init__(self, in_dim, hidden=256, z_dim=128, depth=3, p=0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p)]
            d = hidden
        layers += [nn.Linear(d, z_dim)]
        self.net = nn.Sequential(*layers)
        self.is_sequence = False

    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):
    """
    Transformer-over-event-stream encoder: patient observation SEQUENCE → z.
    Consumes the token sequence from featurize_seq.py.

    Each token = (modality, feat_id, value, t_hours). We embed:
      - modality (symptom/lab/vital/drug) via a small table
      - feat_id  via a per-modality embedding table
      - value    (for labs/vitals) via a linear projection
      - time     via a sinusoidal-ish learned time embedding
    A [CLS] token is prepended; its output is the patient embedding z.
    Drop-in for Encoder: same z_dim output, but forward() takes the batched
    token tensors instead of a flat vector.
    """
    def __init__(self, n_symptoms, n_labs, n_vitals, n_drug_buckets,
                 n_modalities, z_dim=128, d_model=128, n_heads=4, depth=3, p=0.1):
        super().__init__()
        self.is_sequence = True
        self.d_model = d_model
        # one embedding table per modality's feat_id space (padded to max)
        self.sym_emb = nn.Embedding(n_symptoms + 1, d_model)
        self.lab_emb = nn.Embedding(n_labs + 1, d_model)
        self.vit_emb = nn.Embedding(n_vitals + 1, d_model)
        self.drug_emb = nn.Embedding(n_drug_buckets + 1, d_model)
        self.mod_emb = nn.Embedding(n_modalities, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.time_proj = nn.Linear(1, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=p, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Linear(d_model, z_dim)
        # modality constants (kept in sync with featurize_seq)
        self.MOD_SYMPTOM, self.MOD_LAB, self.MOD_VITAL, self.MOD_DRUG = 1, 2, 3, 4

    def forward(self, batch):
        """
        batch: dict of padded tensors, all shape (B, L):
          modality, feat_id  (long);  value, t_hours (float);  pad_mask (bool, True=pad)
        """
        modality = batch["modality"]
        feat_id  = batch["feat_id"]
        value    = batch["value"].unsqueeze(-1)
        t_hours  = batch["t_hours"].unsqueeze(-1)
        pad_mask = batch["pad_mask"]                       # (B, L) True where pad
        B, L = modality.shape

        # per-modality feat embedding (select the right table per token)
        fe = torch.zeros(B, L, self.d_model, device=modality.device)
        for mod, table in ((self.MOD_SYMPTOM, self.sym_emb),
                           (self.MOD_LAB, self.lab_emb),
                           (self.MOD_VITAL, self.vit_emb),
                           (self.MOD_DRUG, self.drug_emb)):
            m = (modality == mod)
            if m.any():
                ids = feat_id.clamp(min=0) * m  # zero out non-matching
                fe = fe + table(ids) * m.unsqueeze(-1)

        tok = (fe
               + self.mod_emb(modality)
               + self.value_proj(value)
               + self.time_proj(torch.log1p(t_hours.clamp(min=0)) / 5.0))

        # prepend CLS
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tok], dim=1)                   # (B, L+1, d)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=modality.device)
        key_pad = torch.cat([cls_pad, pad_mask], dim=1)    # (B, L+1)

        h = self.transformer(x, src_key_padding_mask=key_pad)
        return self.out(h[:, 0])                            # CLS → z


class Predictor(nn.Module):
    """z_context + time-delta → predicted z_target."""
    def __init__(self, z_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z, delta_h):
        # delta_h: (B,1) hours, log-scaled
        d = torch.log1p(delta_h.clamp(min=0)) / 5.0
        return self.net(torch.cat([z, d], dim=-1))


class StateHead(nn.Module):
    """Decode z → per-state signed value (for the supervised anchor + probes)."""
    def __init__(self, z_dim, n_states):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, z_dim), nn.GELU(),
                                 nn.Linear(z_dim, n_states), nn.Tanh())

    def forward(self, z):
        return self.net(z)  # (B, n_states) in [-1,1]


class MedicalJEPA(nn.Module):
    def __init__(self, n_states, z_dim=128, encoder="mlp",
                 in_dim=None, seq_dims=None):
        """
        encoder: "mlp" (needs in_dim) | "transformer" (needs seq_dims dict with
                 n_symptoms/n_labs/n_vitals/n_drug_buckets/n_modalities).
        Training loop, predictor, state head, EMA are identical either way.
        """
        super().__init__()
        self.encoder_type = encoder

        def _make():
            if encoder == "mlp":
                assert in_dim is not None, "mlp encoder needs in_dim"
                return Encoder(in_dim, z_dim=z_dim)
            elif encoder == "transformer":
                return TransformerEncoder(z_dim=z_dim, **seq_dims)
            elif encoder == "ctm":
                return CTMEventEncoder(z_dim=z_dim, **seq_dims)
            else:
                raise ValueError(f"unknown encoder: {encoder}")

        self.context_encoder = _make()
        self.target_encoder = _make()
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor(z_dim)
        self.state_head = StateHead(z_dim, n_states)

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for pt, pc in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            pt.data.mul_(momentum).add_(pc.data, alpha=1 - momentum)


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def _finite_tensor(data, dtype, device):
    t = torch.tensor(data, dtype=dtype, device=device)
    if dtype.is_floating_point:
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def enrich_current_state_examples(examples):
    """
    Add context/current state labels to future examples. Older featurizers only
    stored target_values; dynamics training needs current_values too so it can
    learn target-current deltas.
    """
    lookup = {}
    for e in examples:
        if float(e.get("delta_h", 0.0)) == 0.0:
            lookup[(e["patient_id"], e["t_context"])] = (
                e["target_values"], e["target_mask"], e.get("target_conf", e["target_mask"])
            )
    for e in examples:
        if "context_values" not in e:
            cv = lookup.get((e["patient_id"], e["t_context"]))
            if cv is None:
                # fall back to target labels; delta loss will be masked out if no overlap
                cv = (e["target_values"], e["target_mask"], e.get("target_conf", e["target_mask"]))
            e["context_values"], e["context_mask"], e["context_conf"] = cv
        if "target_context" not in e:
            e["target_context"] = e.get("context")
    return examples


def make_batches(examples, batch_size, device, shuffle=True):
    """MLP path: flat context vectors, target-context vectors, current/target states."""
    idx = list(range(len(examples)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [examples[j] for j in idx[i:i + batch_size]]
        ctx = _finite_tensor([e["context"] for e in chunk], torch.float32, device)
        tctx = _finite_tensor([e.get("target_context", e["context"]) for e in chunk], torch.float32, device)
        tgt = _finite_tensor([e["target_values"] for e in chunk], torch.float32, device)
        msk = _finite_tensor([e["target_mask"] for e in chunk], torch.float32, device)
        cnf = _finite_tensor([e["target_conf"] for e in chunk], torch.float32, device)
        cur = _finite_tensor([e["context_values"] for e in chunk], torch.float32, device)
        cmsk = _finite_tensor([e["context_mask"] for e in chunk], torch.float32, device)
        ccnf = _finite_tensor([e["context_conf"] for e in chunk], torch.float32, device)
        dlt = _finite_tensor([[e["delta_h"]] for e in chunk], torch.float32, device)
        yield ctx, tctx, tgt, msk, cnf, cur, cmsk, ccnf, dlt


def _pad_seq_batch(chunk, key, device):
    L = max(1, max(len(e.get(key, e.get("seq", []))) for e in chunk))
    B = len(chunk)
    modality = torch.zeros(B, L, dtype=torch.long, device=device)
    feat_id  = torch.zeros(B, L, dtype=torch.long, device=device)
    value    = torch.zeros(B, L, dtype=torch.float32, device=device)
    t_hours  = torch.zeros(B, L, dtype=torch.float32, device=device)
    pad_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    for bi, e in enumerate(chunk):
        seq = e.get(key, e.get("seq", [])) or []
        for ti, tk in enumerate(seq[:L]):
            modality[bi, ti] = int(tk.get("modality", 0) or 0)
            feat_id[bi, ti]  = int(tk.get("feat_id", 0) or 0)
            value[bi, ti]    = float(tk.get("value", 0.0) or 0.0)
            t_hours[bi, ti]  = float(tk.get("t_hours", 0.0) or 0.0)
            pad_mask[bi, ti] = False
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    t_hours = torch.nan_to_num(t_hours, nan=0.0, posinf=0.0, neginf=0.0)
    return {"modality": modality, "feat_id": feat_id,
            "value": value, "t_hours": t_hours, "pad_mask": pad_mask}


def make_seq_batches(examples, batch_size, device, shuffle=True):
    """Transformer path: padded token sequences plus current/target states."""
    idx = list(range(len(examples)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [examples[j] for j in idx[i:i + batch_size]]
        batch_in = _pad_seq_batch(chunk, "seq", device)
        # If featurize_seq later provides target_seq, use it for the EMA target encoder.
        # Otherwise this falls back to seq; delta/state losses still train dynamics.
        batch_tgt_in = _pad_seq_batch(chunk, "target_seq", device)
        tgt = _finite_tensor([e["target_values"] for e in chunk], torch.float32, device)
        msk = _finite_tensor([e["target_mask"] for e in chunk], torch.float32, device)
        cnf = _finite_tensor([e["target_conf"] for e in chunk], torch.float32, device)
        cur = _finite_tensor([e["context_values"] for e in chunk], torch.float32, device)
        cmsk = _finite_tensor([e["context_mask"] for e in chunk], torch.float32, device)
        ccnf = _finite_tensor([e["context_conf"] for e in chunk], torch.float32, device)
        dlt = _finite_tensor([[e["delta_h"]] for e in chunk], torch.float32, device)
        yield batch_in, batch_tgt_in, tgt, msk, cnf, cur, cmsk, ccnf, dlt


def masked_mse(pred, target, weight):
    num = ((pred - target) ** 2 * weight).sum()
    den = weight.sum().clamp(min=1.0)
    return num / den

# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_jepa] device = {device} | encoder = {args.encoder}")

    # ── data path depends on encoder ──
    if args.encoder in ("transformer", "ctm"):
        from featurize_seq import build_seq_examples, SeqFeaturizer, N_MODALITIES, N_DRUG_BUCKETS
        examples, vocab = build_seq_examples(args.data)
        examples = enrich_current_state_examples(examples)
        fz = SeqFeaturizer(vocab)
        seq_dims = dict(n_symptoms=fz.n_sym, n_labs=fz.n_lab, n_vitals=fz.n_vit,
                        n_drug_buckets=N_DRUG_BUCKETS, n_modalities=N_MODALITIES)
        in_dim = None
        batcher = make_seq_batches
    else:
        examples, vocab = build_examples(args.data)
        examples = enrich_current_state_examples(examples)
        in_dim = len(examples[0]["context"])
        seq_dims = None
        batcher = make_batches

    n_states = len(vocab["states"])
    obs_dims = torch.tensor(
        [1.0 if e["observability"] in ("lab", "vital") else 0.0 for e in vocab["states"]],
        device=device)
    print(f"[train_jepa] {len(examples)} examples | latent={n_states} "
          f"| observable_dims={int(obs_dims.sum())}")

    model = MedicalJEPA(n_states, z_dim=args.z_dim, encoder=args.encoder,
                        in_dim=in_dim, seq_dims=seq_dims).to(device)
    opt = torch.optim.AdamW(
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters()) +
        list(model.state_head.parameters()), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_jepa = tot_state = n = 0
        for ctx, tctx, tgt, msk, cnf, cur, cmsk, ccnf, dlt in batcher(examples, args.batch_size, device):
            z_ctx = model.context_encoder(ctx)
            with torch.no_grad():
                z_tgt = model.target_encoder(tctx)
            z_hat = model.predictor(z_ctx, dlt)
            jepa_loss = (1 - F.cosine_similarity(z_hat, z_tgt.detach(), dim=-1)).mean()

            pred_current = model.state_head(z_ctx)
            pred_future = model.state_head(z_hat)

            future_ex = (dlt.squeeze(-1) > 0).float().unsqueeze(-1)
            # Same-time examples supervise current state; future examples supervise predicted future state.
            pred_for_target = pred_current * (1.0 - future_ex) + pred_future * future_ex
            target_w = msk * cnf * (1.0 + 2.0 * obs_dims) * (1.0 + future_ex * (args.future_weight - 1.0))
            state_loss = masked_mse(pred_for_target, tgt, target_w)

            # Always anchor the current state when labels exist.
            current_w = cmsk * ccnf * (1.0 + 2.0 * obs_dims)
            current_loss = masked_mse(pred_current, cur, current_w)

            # Explicit dynamics objective: predict delta = future_state - current_state
            # on overlapping observable dims, focused on states that actually changed.
            delta_true = tgt - cur
            delta_pred = pred_future - pred_current
            changed = (delta_true.abs() >= args.change_thresh).float()
            delta_w = msk * cmsk * cnf * (1.0 + 2.0 * obs_dims) * future_ex * changed
            delta_loss = masked_mse(delta_pred, delta_true, delta_w) if delta_w.sum() > 0 else torch.zeros((), device=device)

            loss = (args.jepa_weight * jepa_loss
                    + args.state_weight * state_loss
                    + args.current_weight * current_loss
                    + args.delta_weight * delta_loss)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            model.update_target(args.ema)

            bs = tgt.size(0)
            tot_jepa += jepa_loss.item() * bs
            tot_state += (state_loss.item() + current_loss.item() + delta_loss.item()) * bs
            n += bs
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  jepa={tot_jepa/n:.4f}  state={tot_state/n:.4f}")

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"medical_jepa_{args.encoder}.pt")
    torch.save({"model": model.state_dict(), "vocab": vocab,
                "encoder": args.encoder, "in_dim": in_dim, "seq_dims": seq_dims,
                "n_states": n_states, "z_dim": args.z_dim}, ckpt)
    print(f"[train_jepa] saved checkpoint: {ckpt}")
    print("[train_jepa] NOTE: this dynamics-aware version adds future-state and delta-state losses. "
          "Clinical claims still require real EHR validation and held-out outcomes.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "osler_coverage_trajectories.jsonl"))
    ap.add_argument("--out", default=os.path.join(_HERE, "checkpoints"))
    ap.add_argument("--encoder", choices=["mlp", "transformer", "ctm"], default="mlp",
                    help="mlp = flat context vector; transformer = event-stream sequence")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--jepa_weight", type=float, default=1.0)
    ap.add_argument("--state_weight", type=float, default=1.0)
    ap.add_argument("--current_weight", type=float, default=0.25,
                    help="extra anchor on current-state labels")
    ap.add_argument("--future_weight", type=float, default=3.0,
                    help="up-weight future examples in supervised state loss")
    ap.add_argument("--delta_weight", type=float, default=2.0,
                    help="weight for explicit future-current delta-state loss")
    ap.add_argument("--change_thresh", type=float, default=0.1,
                    help="minimum |future-current| to include in delta loss")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())
