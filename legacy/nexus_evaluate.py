"""
NEXUS Medical Agent — Evaluation Suite
═══════════════════════════════════════════════════════════════
Runs after training to produce:
  1. Save model checkpoint (weights + config + vocab)
  2. Independent test set evaluation (epsilon=0, no updates)
  3. Confusion matrix (disease prediction)
  4. Per-class metrics (precision, recall, F1, support)
  5. High-risk disease recall (heart attack, meningitis, sepsis)
  6. Multi-seed stability report

Usage:
    # After training completes:
    python nexus_engine/nexus_evaluate.py

    # With a saved checkpoint:
    python nexus_engine/nexus_evaluate.py --checkpoint nexus_checkpoint.npz
"""
import sys, os, argparse
_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
for _p in [_here, _parent]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json, time, random
import numpy as np
from collections import defaultdict

# ── Suppress [ANATOMY] spam ──────────────────────────────────
import builtins as _bi
_real_print = _bi.print
_anatomy_seen = [False]
def _quiet(*args, **kw):
    msg = " ".join(str(a) for a in args)
    if msg.startswith("[ANATOMY]"):
        if not _anatomy_seen[0]:
            _anatomy_seen[0] = True
            _real_print(*args, **kw)
        return
    _real_print(*args, **kw)
_bi.print = _quiet

# ── Load modules ──────────────────────────────────────────────
try:
    from nexus_engine.nexus_medical import NexusMedical
    from nexus_engine.nexus_learning_bridge import NexusLearner
except ModuleNotFoundError:
    from nexus_medical import NexusMedical
    from nexus_learning_bridge import NexusLearner

# Load env — find the copy that has _DISEASES (the updated version)
import importlib.util, shutil

def _load_env_module(path):
    spec = importlib.util.spec_from_file_location("nexus_learning_env", path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["nexus_learning_env"] = mod
    spec.loader.exec_module(mod)
    return mod

_candidates = [
    os.path.join(_here,   "nexus_learning_env.py"),   # nexus_engine/
    os.path.join(_parent, "nexus_learning_env.py"),   # project root
]

_mod = None
for _cand in _candidates:
    if not os.path.exists(_cand):
        continue
    try:
        _m = _load_env_module(_cand)
        if hasattr(_m, "_DISEASES"):
            _mod = _m
            _real_print(f"[EVAL] Loaded env from: {_cand}")
            break
        else:
            _real_print(f"[EVAL] Skipping {_cand} (old version, no _DISEASES)")
    except Exception as _e:
        _real_print(f"[EVAL] Could not load {_cand}: {_e}")

if _mod is None:
    raise ImportError(
        "Could not find an updated nexus_learning_env.py with _DISEASES.\n"
        "Please download the latest nexus_learning_env.py from Claude "
        "and place it in:\n"
        f"  {_here}/nexus_learning_env.py\n"
        f"or {_parent}/nexus_learning_env.py"
    )

MedicalEnv          = _mod.MedicalEnv
NexusRLAgent        = _mod.NexusRLAgent
PATIENT_POOL        = _mod.PATIENT_POOL
TREATMENT_OPTIONS   = _mod.TREATMENT_OPTIONS
_DISEASES           = _mod._DISEASES
_TREATMENTS         = _mod._TREATMENTS
_action_idx         = _mod._action_idx
_idx_to_action      = _mod._idx_to_action
_build_feature_vector = _mod._build_feature_vector
train               = _mod.train

SEP = "═" * 62
HIGH_RISK = {"heart attack", "meningitis", "sepsis"}


# ════════════════════════════════════════════════════════════════
# Checkpoint save / load
# ════════════════════════════════════════════════════════════════

def save_checkpoint(agent, path="nexus_checkpoint.npz"):
    clf = agent._clf
    np.savez(path,
        W1=clf.W1, b1=clf.b1,
        Wdx=clf.Wdx, bdx=clf.bdx,
        Wtx=clf.Wtx, btx=clf.btx,
        diseases=np.array(_DISEASES),
        treatments=np.array(_TREATMENTS),
        epsilon=np.array([agent.epsilon]),
        episode=np.array([agent._episode]),
    )
    print(f"[CKPT] Saved → {path}  ({os.path.getsize(path)//1024} KB)")


def load_checkpoint(agent, path="nexus_checkpoint.npz"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    d = np.load(path, allow_pickle=True)
    clf = agent._clf
    clf.W1[:] = d["W1"]; clf.b1[:] = d["b1"]
    clf.Wdx[:] = d["Wdx"]; clf.bdx[:] = d["bdx"]
    clf.Wtx[:] = d["Wtx"]; clf.btx[:] = d["btx"]
    agent.epsilon  = float(d["epsilon"][0])
    agent._episode = int(d["episode"][0])
    print(f"[CKPT] Loaded ← {path}  (ep {agent._episode}, ε={agent.epsilon:.3f})")
    return agent


# ════════════════════════════════════════════════════════════════
# Test set evaluation
# ════════════════════════════════════════════════════════════════

def evaluate(agent, nexus, n_episodes=500, seed=42, noise_p=0.15):
    """
    Pure test evaluation — epsilon=0, no weight updates, fixed seed.
    Returns dict of metrics.
    """
    random.seed(seed)
    np.random.seed(seed)

    env = MedicalEnv(nexus, noise_p=noise_p)
    saved_eps    = agent.epsilon
    agent.epsilon = 0.0   # pure exploitation

    y_true_dx, y_pred_dx = [], []
    y_true_tx, y_pred_tx = [], []
    rewards = []
    per_disease = defaultdict(lambda: {"correct_dx": 0, "correct_tx": 0, "total": 0})

    for _ in range(n_episodes):
        obs       = env.reset()
        state_vec = agent.encode_state(obs)
        dx, tx    = agent.choose_action(state_vec)
        patient   = env._current_patient
        nexus_result = env._current_nexus_result

        true_dx = patient["disease"]
        correct_dx = env._matches(dx, true_dx)
        correct_tx = any(env._matches(tx, t) for t in patient["correct_treatments"])
        reward     = env._compute_reward(dx, tx, nexus_result)

        y_true_dx.append(true_dx)
        y_pred_dx.append(dx)
        y_true_tx.append(patient["correct_treatments"][0])
        y_pred_tx.append(tx)
        rewards.append(reward)

        per_disease[true_dx]["total"]      += 1
        per_disease[true_dx]["correct_dx"] += int(correct_dx)
        per_disease[true_dx]["correct_tx"] += int(correct_tx)

    agent.epsilon = saved_eps

    dx_acc = sum(a == b for a, b in zip(y_true_dx, y_pred_dx)) / n_episodes
    tx_acc = sum(env._matches(a, b) for a, b in zip(y_true_tx, y_pred_tx)) / n_episodes

    return {
        "dx_acc":      dx_acc,
        "tx_acc":      tx_acc,
        "avg_reward":  sum(rewards) / len(rewards),
        "n_episodes":  n_episodes,
        "y_true_dx":   y_true_dx,
        "y_pred_dx":   y_pred_dx,
        "per_disease": dict(per_disease),
    }


# ════════════════════════════════════════════════════════════════
# Confusion matrix
# ════════════════════════════════════════════════════════════════

def print_confusion_matrix(y_true, y_pred, labels):
    n = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        ti = label_idx.get(t, -1)
        pi = label_idx.get(p, -1)
        if ti >= 0 and pi >= 0:
            matrix[ti][pi] += 1

    # Short names for display
    short = [l[:10] for l in labels]
    col_w = 11

    print(f"\n{'Confusion Matrix (rows=true, cols=pred)':^{col_w * (n+1)}}")
    print(" " * 12 + "".join(f"{s:>{col_w}}" for s in short))
    print(" " * 12 + "─" * (col_w * n))
    for i, label in enumerate(labels):
        row_str = f"{short[i]:>11} │"
        for j in range(n):
            val = matrix[i][j]
            marker = "◀" if i == j else " "
            row_str += f"{val:>{col_w-1}}{marker}"
        print(row_str)

    # Per-class recall from matrix diagonal
    print()
    for i, label in enumerate(labels):
        total = matrix[i].sum()
        correct = matrix[i][i]
        recall = correct / total if total > 0 else 0
        bar = "█" * int(recall * 20) + "░" * (20 - int(recall * 20))
        risk = " ⚠ HIGH RISK" if label in HIGH_RISK else ""
        print(f"  {label:20s} recall={recall:5.1%}  {bar}{risk}")


# ════════════════════════════════════════════════════════════════
# Per-class precision / recall / F1
# ════════════════════════════════════════════════════════════════

def print_per_class_metrics(y_true, y_pred, labels):
    print(f"\n{'Disease':22s} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}")
    print("─" * 50)

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        sup = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        risk = " ⚠" if label in HIGH_RISK else "  "
        alert = " ← LOW RECALL" if rec < 0.5 and label in HIGH_RISK else ""
        print(f"  {risk}{label:20s} {prec:6.1%} {rec:6.1%} {f1:6.1%} {sup:5d}{alert}")


# ════════════════════════════════════════════════════════════════
# Multi-seed stability test
# ════════════════════════════════════════════════════════════════

def multi_seed_test(nexus, seeds=(42, 123, 999), episodes=2000):
    print(f"\n{SEP}")
    print(f"Multi-seed stability test  ({len(seeds)} seeds × {episodes} episodes)")
    print(SEP)

    results = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        print(f"\n  Seed {seed}: training {episodes} episodes...")

        agent, _, _, _, env = train(
            episodes=episodes,
            print_every=episodes + 1,  # suppress per-episode output
            learn_every=200,
            verbose=False,
        )

        metrics = evaluate(agent, nexus, n_episodes=300, seed=seed)
        results.append({"seed": seed, **metrics})
        print(f"  Seed {seed}: Dx={metrics['dx_acc']:.1%}  "
              f"Tx={metrics['tx_acc']:.1%}  "
              f"reward={metrics['avg_reward']:+.3f}")

    dx_vals = [r["dx_acc"] for r in results]
    tx_vals = [r["tx_acc"] for r in results]
    print(f"\n  Dx: mean={np.mean(dx_vals):.1%}  std={np.std(dx_vals):.1%}  "
          f"min={min(dx_vals):.1%}  max={max(dx_vals):.1%}")
    print(f"  Tx: mean={np.mean(tx_vals):.1%}  std={np.std(tx_vals):.1%}  "
          f"min={min(tx_vals):.1%}  max={max(tx_vals):.1%}")

    if np.std(dx_vals) < 0.05:
        print("  RESULT: ✓ Stable — std < 5%")
    else:
        print("  RESULT: ⚠ Unstable — std ≥ 5%, check training consistency")

    return results


# ════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Path to .npz checkpoint")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--test-n", type=int, default=500)
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("NEXUS Medical Agent — Evaluation Suite")
    print(SEP)

    print("\nLoading NEXUS knowledge web...")
    nexus = NexusMedical()
    nexus.load_knowledge()

    if args.checkpoint:
        # Load from checkpoint
        agent = NexusRLAgent()
        env   = MedicalEnv(nexus, noise_p=0.15)
        load_checkpoint(agent, args.checkpoint)
    else:
        # Train fresh
        print(f"\nTraining {args.episodes} episodes...")
        agent, nexus, learner, results, env = train(
            episodes=args.episodes,
            print_every=50,
            learn_every=200,
            verbose=True,
        )
        # Save checkpoint immediately
        save_checkpoint(agent, "nexus_checkpoint.npz")

    # ── Test set evaluation ──────────────────────────────────
    print(f"\n{SEP}")
    print(f"Independent test evaluation  (n={args.test_n}, ε=0, seed=42)")
    print(SEP)

    metrics = evaluate(agent, nexus, n_episodes=args.test_n, seed=42)
    print(f"\n  Test Dx accuracy : {metrics['dx_acc']:.1%}")
    print(f"  Test Tx accuracy : {metrics['tx_acc']:.1%}")
    print(f"  Test avg reward  : {metrics['avg_reward']:+.3f}")

    # ── Confusion matrix ─────────────────────────────────────
    print(f"\n{SEP}")
    print("Confusion matrix")
    print(SEP)
    print_confusion_matrix(
        metrics["y_true_dx"], metrics["y_pred_dx"], _DISEASES
    )

    # ── Per-class metrics ────────────────────────────────────
    print(f"\n{SEP}")
    print("Per-class precision / recall / F1")
    print(SEP)
    print_per_class_metrics(
        metrics["y_true_dx"], metrics["y_pred_dx"], _DISEASES
    )

    # ── High-risk diseases ───────────────────────────────────
    print(f"\n{SEP}")
    print("High-risk disease recall")
    print(SEP)
    all_ok = True
    for disease in HIGH_RISK:
        d = metrics["per_disease"].get(disease, {})
        total  = d.get("total", 0)
        recall = d.get("correct_dx", 0) / total if total > 0 else 0.0
        status = "✓" if recall >= 0.7 else "✗ UNSAFE"
        if recall < 0.7:
            all_ok = False
        print(f"  {status}  {disease:20s}  recall={recall:.1%}  ({d.get('correct_dx',0)}/{total})")

    if all_ok:
        print("\n  All high-risk diseases above 70% recall threshold ✓")
    else:
        print("\n  ⚠ Some high-risk diseases below 70% — needs improvement")

    # ── Per-disease breakdown ────────────────────────────────
    print(f"\n{SEP}")
    print("Per-disease accuracy breakdown")
    print(SEP)
    for disease, d in sorted(metrics["per_disease"].items()):
        total = d["total"]
        dx_r  = d["correct_dx"] / total if total > 0 else 0
        tx_r  = d["correct_tx"] / total if total > 0 else 0
        bar   = "█" * int(dx_r * 15) + "░" * (15 - int(dx_r * 15))
        risk  = " ⚠" if disease in HIGH_RISK else "  "
        print(f"  {risk}{disease:20s}  Dx={dx_r:5.1%}  Tx={tx_r:5.1%}  {bar}  (n={total})")

    # ── Multi-seed stability ─────────────────────────────────
    if args.multi_seed:
        multi_seed_test(nexus, seeds=(42, 123, 999), episodes=args.episodes)

    # Save eval report
    report = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episodes":    args.episodes,
        "test_n":      args.test_n,
        "dx_acc":      metrics["dx_acc"],
        "tx_acc":      metrics["tx_acc"],
        "avg_reward":  metrics["avg_reward"],
        "per_disease": {
            k: {
                "dx_recall": v["correct_dx"] / v["total"] if v["total"] > 0 else 0,
                "tx_recall": v["correct_tx"] / v["total"] if v["total"] > 0 else 0,
                "support":   v["total"],
            }
            for k, v in metrics["per_disease"].items()
        }
    }
    with open("nexus_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{SEP}")
    print(f"Report saved → nexus_eval_report.json")
    print(SEP)


if __name__ == "__main__":
    main()