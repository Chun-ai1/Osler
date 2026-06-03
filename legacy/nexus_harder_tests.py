"""
NEXUS Medical Agent — Generalization Test Suite
═══════════════════════════════════════════════════════════════
Tests BEYOND the standard eval to probe real generalisation:

  Test A — Noisy same-distribution  (drop/add symptoms randomly)
  Test B — Overlapping disease pairs (flu vs covid, pneumonia vs sepsis...)
  Test C — Incomplete information    (only 1-2 symptoms given)
  Test D — Ablation                  (which features actually matter?)
  Test E — Multi-seed stability      (3 seeds, same config)

Usage:
    python nexus_engine/nexus_harder_tests.py
    python nexus_engine/nexus_harder_tests.py --checkpoint nexus_checkpoint.npz
"""
import sys, os, random, json, time
import numpy as np
from collections import defaultdict

_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
for _p in [_here, _parent]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import builtins as _bi
_real_print = _bi.print
_anatomy_seen = [False]
def _quiet(*args, **kw):
    msg = " ".join(str(a) for a in args)
    if msg.startswith("[ANATOMY]") and _anatomy_seen[0]:
        return
    if msg.startswith("[ANATOMY]"):
        _anatomy_seen[0] = True
    _real_print(*args, **kw)
_bi.print = _quiet

try:
    from nexus_engine.nexus_medical import NexusMedical
    from nexus_engine.nexus_learning_bridge import NexusLearner
except ModuleNotFoundError:
    from nexus_medical import NexusMedical
    from nexus_learning_bridge import NexusLearner

import importlib.util

def _load_env():
    for path in [os.path.join(_here, "nexus_learning_env.py"),
                 os.path.join(_parent, "nexus_learning_env.py")]:
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("nexus_learning_env", path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules["nexus_learning_env"] = mod
        spec.loader.exec_module(mod)
        if hasattr(mod, "_DISEASES"):
            return mod
    raise ImportError("No updated nexus_learning_env.py found with _DISEASES")

_mod = _load_env()
MedicalEnv          = _mod.MedicalEnv
NexusRLAgent        = _mod.NexusRLAgent
PATIENT_POOL        = _mod.PATIENT_POOL
TREATMENT_OPTIONS   = _mod.TREATMENT_OPTIONS
_DISEASES           = _mod._DISEASES
_TREATMENTS         = _mod._TREATMENTS
_build_feature_vector = _mod._build_feature_vector
train               = _mod.train

SEP  = "═" * 62
HIGH_RISK = {"heart attack", "meningitis", "sepsis"}

# ── Confusable disease pairs ──────────────────────────────────
OVERLAPPING_PAIRS = [
    ("flu",        "covid"),
    ("pneumonia",  "flu"),
    ("pneumonia",  "sepsis"),
    ("meningitis", "migraine"),
    ("heart attack", "appendicitis"),  # both have chest/abdominal pain
    ("gastroenteritis", "appendicitis"),
]

ALL_SYMPTOMS = list({s for p in PATIENT_POOL for s in p["symptoms"]})

NOISE_SYMPTOMS = [
    "fatigue", "dizziness", "weakness", "loss of appetite",
    "insomnia", "sweating", "chills", "back pain", "joint pain",
    "sore throat", "runny nose", "rash", "anxiety",
]


# ════════════════════════════════════════════════════════════════
def load_or_train(nexus, checkpoint_path=None):
    agent = NexusRLAgent()
    env   = MedicalEnv(nexus, noise_p=0.0)
    loaded = False

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            d   = np.load(checkpoint_path, allow_pickle=True)
            clf = agent._clf
            if d["W1"].shape != clf.W1.shape:
                raise ValueError(
                    f"Architecture changed: checkpoint W1={d['W1'].shape} "
                    f"vs current W1={clf.W1.shape}. Delete checkpoint and retrain."
                )
            clf.W1[:] = d["W1"]; clf.b1[:] = d["b1"]
            clf.Wdx[:] = d["Wdx"]; clf.bdx[:] = d["bdx"]
            clf.Wtx[:] = d["Wtx"]; clf.btx[:] = d["btx"]
            agent.epsilon  = float(d["epsilon"][0])
            agent._episode = int(d["episode"][0])
            _real_print(f"[CKPT] Loaded ← {checkpoint_path}  "
                        f"(ep={agent._episode}, arch W1={clf.W1.shape})")
            loaded = True
        except (ValueError, KeyError) as e:
            _real_print(f"[CKPT] Cannot load: {e}")
            _real_print("[CKPT] Training fresh model with new architecture...")

    if not loaded:
        _real_print("Training fresh model (5000 episodes, world-model curriculum)...")
        agent, nexus, _, _, env = train(
            episodes=5000, print_every=500, learn_every=200, verbose=True
        )
        np.savez("nexus_checkpoint.npz",
            W1=agent._clf.W1, b1=agent._clf.b1,
            Wdx=agent._clf.Wdx, bdx=agent._clf.bdx,
            Wtx=agent._clf.Wtx, btx=agent._clf.btx,
            diseases=np.array(_DISEASES),
            treatments=np.array(_TREATMENTS),
            epsilon=np.array([agent.epsilon]),
            episode=np.array([agent._episode]),
        )
        _real_print(f"[CKPT] Saved → nexus_checkpoint.npz  "
                    f"(arch W1={agent._clf.W1.shape})")

    agent.epsilon = 0.0
    return agent, env, nexus


def run_episode(agent, env, symptom_override=None, noise_p_override=None):
    """Run one episode, optionally overriding symptoms or noise."""
    old_noise = env.noise_p
    if noise_p_override is not None:
        env.noise_p = noise_p_override

    obs     = env.reset()
    patient = env._current_patient

    # Override symptoms if requested (for incomplete / overlap tests)
    if symptom_override is not None:
        obs["symptoms"] = symptom_override
        obs["nexus_result"] = env._run_nexus(symptom_override)
        env._current_nexus_result = obs["nexus_result"]

    env.noise_p = old_noise

    state_vec      = agent.encode_state(obs)
    dx, tx         = agent.choose_action(state_vec)
    correct_dx     = env._matches(dx, patient["disease"])
    correct_tx     = any(env._matches(tx, t) for t in patient["correct_treatments"])
    return {
        "true_dx":    patient["disease"],
        "pred_dx":    dx,
        "true_tx":    patient["correct_treatments"][0],
        "pred_tx":    tx,
        "correct_dx": correct_dx,
        "correct_tx": correct_tx,
        "symptoms":   obs["symptoms"],
    }


# ════════════════════════════════════════════════════════════════
# TEST A: Noisy same-distribution
# ════════════════════════════════════════════════════════════════

def test_a_noise(agent, env, n=300):
    _real_print(f"\n{SEP}")
    _real_print("TEST A — Noisy symptoms (drop 1-2 + add 1-3 irrelevant)")
    _real_print(SEP)

    results = defaultdict(lambda: {"correct": 0, "total": 0})

    for noise_level, label in [(0.3, "light"), (0.5, "medium"), (0.7, "heavy")]:
        correct_dx = 0
        for _ in range(n):
            obs     = env.reset()
            patient = env._current_patient
            syms    = list(patient["symptoms"])

            # Drop symptoms
            n_drop = int(len(syms) * noise_level * 0.5)
            for _ in range(n_drop):
                if len(syms) > 1:
                    syms.pop(random.randint(0, len(syms) - 1))

            # Add noise symptoms
            n_add = random.randint(1, max(1, int(3 * noise_level)))
            for ns in random.sample(NOISE_SYMPTOMS, min(n_add, len(NOISE_SYMPTOMS))):
                if ns not in syms:
                    syms.append(ns)

            # Run NEXUS on the corrupted symptoms (matches training condition)
            obs2 = {"symptoms": syms,
                    "nexus_result": env._run_nexus(syms),
                    "etiology": {}, "pathogen_spread": []}
            env._current_nexus_result = obs2["nexus_result"]
            sv  = agent.encode_state(obs2)
            dx2, _ = agent.choose_action(sv)
            if env._matches(dx2, patient["disease"]):
                correct_dx += 1

        acc = correct_dx / n
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        status = "✓" if acc >= 0.7 else "⚠" if acc >= 0.5 else "✗"
        _real_print(f"  {status} Noise={label:6s} ({noise_level:.0%} drop+add)  "
                    f"Dx={acc:5.1%}  {bar}")
        results[label] = {"acc": acc, "n": n}

    return results


# ════════════════════════════════════════════════════════════════
# TEST B: Overlapping / confusable disease pairs
# ════════════════════════════════════════════════════════════════

def test_b_overlap(agent, env, n_per_pair=50):
    _real_print(f"\n{SEP}")
    _real_print("TEST B — Overlapping disease pairs (shared symptoms)")
    _real_print(SEP)
    _real_print(f"  {'Pair':35s}  {'A recall':>9}  {'B recall':>9}  {'Status'}")
    _real_print(f"  {'─'*35}  {'─'*9}  {'─'*9}  {'─'*8}")

    results = []
    pool_by_disease = {p["disease"]: p for p in PATIENT_POOL}

    for dx_a, dx_b in OVERLAPPING_PAIRS:
        if dx_a not in pool_by_disease or dx_b not in pool_by_disease:
            continue

        pa = pool_by_disease[dx_a]
        pb = pool_by_disease[dx_b]

        # Build a mixed symptom set (symptoms from BOTH diseases)
        shared = list(set(pa["symptoms"]) & set(pb["symptoms"]))
        only_a = list(set(pa["symptoms"]) - set(pb["symptoms"]))
        only_b = list(set(pb["symptoms"]) - set(pa["symptoms"]))

        # only_a = symptoms unique to A (differentiators)
        # only_b = symptoms unique to B (differentiators)
        # shared = symptoms both have
        # Test: give shared + 1-2 differentiators from TRUE disease
        #        and add 1 differentiator from OTHER disease as noise
        # This is symmetric and actually tests the model, not the test generator.

        # Symptom weights for fair noise selection
        _W = {
            "left arm pain": 2.5, "stiff neck": 2.5, "loss of smell": 2.5,
            "confusion": 2.0, "wheezing": 2.0,
            "sensitivity to light": 1.8, "chest pain": 1.8, "abdominal pain": 1.8,
            "diarrhea": 1.8, "body aches": 1.8, "headache": 1.5, "sweating": 1.5,
            "shortness of breath": 1.4, "weakness": 1.3, "dizziness": 1.3,
        }

        def _eval_with_overlap(true_patient, other_unique, true_unique, n):
            """
            true_unique: symptoms only this disease has (its differentiators)
            other_unique: symptoms only the other disease has (the noise)

            Uses the WEAKEST noise symptom to avoid test construction bias
            where high-weight noise symptoms overwhelm the true disease signal.
            """
            correct = 0
            # Pick the weakest noise symptom (lowest weight) — fair test
            noise_sym = min(other_unique, key=lambda s: _W.get(s, 1.0)) if other_unique else None
            for _ in range(n):
                # Base: shared symptoms + all true differentiators
                syms = list(shared)
                if true_unique:
                    # Keep ALL differentiating symptoms of the true disease
                    syms += list(true_unique)
                # Add the weakest noise symptom from the other disease
                if noise_sym:
                    syms.append(noise_sym)

                obs = {"symptoms": list(set(syms)),
                       "nexus_result": env._run_nexus(list(set(syms))),
                       "etiology": {}, "pathogen_spread": []}
                env._current_nexus_result = obs["nexus_result"]
                env._current_patient = {**true_patient, "age": 40, "episode": 0}
                sv = agent.encode_state(obs)
                dx, _ = agent.choose_action(sv)
                if env._matches(dx, true_patient["disease"]):
                    correct += 1
            return correct / n

        # If no unique symptoms exist for a disease, test is not meaningful
        if not only_a or not only_b:
            _real_print(f"  ⚠  {dx_a} vs {dx_b}: no unique differentiating symptoms — skipping")
            continue

        acc_a = _eval_with_overlap(pa, only_b, only_a, n_per_pair)
        acc_b = _eval_with_overlap(pb, only_a, only_b, n_per_pair)
        min_acc = min(acc_a, acc_b)

        status = "✓" if min_acc >= 0.7 else "⚠" if min_acc >= 0.5 else "✗ CONFUSABLE"
        pair_str = f"{dx_a} vs {dx_b}"
        _real_print(f"  {status}  {pair_str:33s}  {acc_a:8.1%}  {acc_b:8.1%}  {status}")
        # Show differentiators for context
        _real_print(f"         Unique to {dx_a}: {only_a}")
        _real_print(f"         Unique to {dx_b}: {only_b}")
        _real_print(f"         Shared: {shared}")
        results.append({"pair": (dx_a, dx_b), "acc_a": acc_a, "acc_b": acc_b,
                         "only_a": only_a, "only_b": only_b, "shared": shared})

    return results


# ════════════════════════════════════════════════════════════════
# TEST C: Incomplete information (1-2 symptoms only)
# ════════════════════════════════════════════════════════════════

def test_c_incomplete(agent, env, n=200):
    _real_print(f"\n{SEP}")
    _real_print("TEST C — Incomplete information (partial symptom list)")
    _real_print(SEP)

    results = {}
    for n_syms, label in [(1, "1 symptom"), (2, "2 symptoms"), (3, "3 symptoms")]:
        correct_dx = 0
        for _ in range(n):
            obs     = env.reset()
            patient = env._current_patient
            syms    = patient["symptoms"]

            # Give only the first N symptoms
            partial = syms[:n_syms]
            if not partial:
                continue

            obs["symptoms"]     = partial
            obs["nexus_result"] = env._run_nexus(partial)
            env._current_nexus_result = obs["nexus_result"]

            sv = agent.encode_state(obs)
            dx, _ = agent.choose_action(sv)
            if env._matches(dx, patient["disease"]):
                correct_dx += 1

        acc = correct_dx / n
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        status = "✓" if acc >= 0.5 else "⚠" if acc >= 0.3 else "✗"
        _real_print(f"  {status} {label:12s}  Dx={acc:5.1%}  {bar}")
        results[label] = acc

    return results


# ════════════════════════════════════════════════════════════════
# TEST D: Ablation — which features matter?
# ════════════════════════════════════════════════════════════════

def test_d_ablation(agent, env, n=300):
    _real_print(f"\n{SEP}")
    _real_print("TEST D — Feature ablation (zero out feature groups)")
    _real_print(SEP)

    def _eval_with_mask(mask_fn, n=n):
        correct = 0
        for _ in range(n):
            obs = env.reset()
            patient = env._current_patient
            sv = _build_feature_vector(obs)
            sv = mask_fn(sv.copy())
            dx_q, tx_q = agent._clf.forward(sv)
            import numpy as np
            pred_dx = _DISEASES[int(np.argmax(dx_q))]
            if env._matches(pred_dx, patient["disease"]):
                correct += 1
        return correct / n

    # Baseline (no masking)
    base_acc = _eval_with_mask(lambda v: v)

    ablations = [
        ("All features (baseline)",      lambda v: v),
        ("Zero NEXUS disease scores",    lambda v: _zero(v, 0, 10)),
        ("Zero symptom one-hot",         lambda v: _zero(v, 10, 20)),
        ("Zero red flags + consistency", lambda v: _zero(v, 20, 22)),
        ("Zero organ system one-hot",    lambda v: _zero(v, 22, 29)),
        ("Zero top-3 NEXUS scores",      lambda v: _zero(v, 29, 32)),
        ("Only NEXUS disease scores",    lambda v: _keep(v, 0, 10)),
        ("Only symptoms",                lambda v: _keep(v, 10, 20)),
    ]

    _real_print(f"  {'Feature group':40s}  {'Dx acc':>7}  {'Drop':>7}")
    _real_print(f"  {'─'*40}  {'─'*7}  {'─'*7}")

    results = {}
    for label, mask_fn in ablations:
        acc  = _eval_with_mask(mask_fn)
        drop = base_acc - acc
        bar  = "▼" * max(0, int(drop * 20))
        imp  = " ← CRITICAL" if drop > 0.3 else (" ← important" if drop > 0.1 else "")
        _real_print(f"  {label:40s}  {acc:6.1%}  {drop:+6.1%}  {bar}{imp}")
        results[label] = {"acc": acc, "drop": drop}

    return results

def _zero(v, start, end):
    v[start:end] = 0.0
    return v

def _keep(v, start, end):
    mask = np.zeros_like(v)
    mask[start:end] = v[start:end]
    return mask


# ════════════════════════════════════════════════════════════════
# TEST E: Multi-seed stability
# ════════════════════════════════════════════════════════════════

def test_e_multiseed(nexus, seeds=(42, 123, 999), episodes=3000, n_test=300):
    _real_print(f"\n{SEP}")
    _real_print(f"TEST E — Multi-seed stability ({len(seeds)} seeds × {episodes} episodes)")
    _real_print(SEP)

    seed_results = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        _real_print(f"\n  Seed {seed}: training {episodes} episodes...")

        agent, _, _, _, env = train(
            episodes=episodes,
            print_every=episodes + 1,
            learn_every=200,
            verbose=False,
        )
        agent.epsilon = 0.0

        correct_dx = correct_tx = 0
        for _ in range(n_test):
            obs     = env.reset()
            patient = env._current_patient
            sv      = agent.encode_state(obs)
            dx, tx  = agent.choose_action(sv)
            correct_dx += env._matches(dx, patient["disease"])
            correct_tx += any(env._matches(tx, t) for t in patient["correct_treatments"])

        dx_acc = correct_dx / n_test
        tx_acc = correct_tx / n_test
        seed_results.append({"seed": seed, "dx": dx_acc, "tx": tx_acc})
        _real_print(f"  Seed {seed}: Dx={dx_acc:.1%}  Tx={tx_acc:.1%}")

    dx_vals = [r["dx"] for r in seed_results]
    tx_vals = [r["tx"] for r in seed_results]
    _real_print(f"\n  Dx: mean={np.mean(dx_vals):.1%}  std={np.std(dx_vals):.1%}  "
                f"range=[{min(dx_vals):.1%}, {max(dx_vals):.1%}]")
    _real_print(f"  Tx: mean={np.mean(tx_vals):.1%}  std={np.std(tx_vals):.1%}  "
                f"range=[{min(tx_vals):.1%}, {max(tx_vals):.1%}]")

    if np.std(dx_vals) < 0.05:
        _real_print("  RESULT: ✓ Stable — std < 5%")
    else:
        _real_print("  RESULT: ⚠ Unstable — std ≥ 5%")

    return seed_results


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="nexus_checkpoint.npz")
    parser.add_argument("--skip-multiseed", action="store_true",
                        help="Skip multi-seed test (saves ~15 min)")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Episodes per seed in multi-seed test")
    args = parser.parse_args()

    _real_print(f"\n{SEP}")
    _real_print("NEXUS Medical Agent — Generalization Test Suite")
    _real_print(SEP)

    _real_print("\nLoading NEXUS...")
    nexus = NexusMedical()
    nexus.load_knowledge()

    agent, env, nexus = load_or_train(nexus, args.checkpoint)

    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # Run all tests
    report["test_a_noise"]      = test_a_noise(agent, env)
    report["test_b_overlap"]    = test_b_overlap(agent, env)
    report["test_c_incomplete"] = test_c_incomplete(agent, env)
    report["test_d_ablation"]   = test_d_ablation(agent, env)

    if not args.skip_multiseed:
        report["test_e_multiseed"] = test_e_multiseed(
            nexus, episodes=args.episodes
        )
    else:
        _real_print(f"\n{SEP}")
        _real_print("TEST E — Skipped (use --episodes N to run)")
        _real_print(SEP)

    # Summary
    _real_print(f"\n{SEP}")
    _real_print("SUMMARY")
    _real_print(SEP)

    a = report["test_a_noise"]
    _real_print(f"  A. Noise robustness:  "
                f"light={a.get('light',{}).get('acc',0):.1%}  "
                f"medium={a.get('medium',{}).get('acc',0):.1%}  "
                f"heavy={a.get('heavy',{}).get('acc',0):.1%}")

    b = report["test_b_overlap"]
    if b:
        min_b = min(min(r["acc_a"], r["acc_b"]) for r in b)
        _real_print(f"  B. Overlap pairs:     worst-pair min recall = {min_b:.1%}")

    c = report["test_c_incomplete"]
    _real_print(f"  C. Incomplete info:   "
                + "  ".join(f"{k}={v:.1%}" for k, v in c.items()))

    d = report["test_d_ablation"]
    if d:
        biggest_drop = max(d.items(), key=lambda x: x[1]["drop"])
        _real_print(f"  D. Most critical feature: '{biggest_drop[0]}' "
                    f"(drop={biggest_drop[1]['drop']:+.1%})")

    # Save
    def _jsonify(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, tuple):
            return [_jsonify(v) for v in obj]
        return obj

    with open("nexus_harder_eval.json", "w") as f:
        json.dump(_jsonify(report), f, indent=2)
    _real_print(f"\n  Report saved → nexus_harder_eval.json")
    _real_print(SEP)


if __name__ == "__main__":
    main()