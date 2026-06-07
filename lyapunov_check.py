import argparse
import json
import math
import torch
import numpy as np

from train_jepa import MedicalJEPA, make_seq_batches
from featurize_seq import build_seq_examples


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = MedicalJEPA(
        ck["n_states"],
        z_dim=ck["z_dim"],
        encoder=ck["encoder"],
        in_dim=ck.get("in_dim"),
        seq_dims=ck.get("seq_dims"),
    ).to(device)

    model.load_state_dict(ck["model"])
    model.eval()

    return model, ck


def safe_float(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--delta_h", type=float, default=24.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ck = load_model(args.ckpt, device)
    examples, vocab = build_seq_examples(args.data)

    print(f"[lyapunov] encoder={ck['encoder']} examples={len(examples)} device={device}")

    delta_vs = []
    cur_norms = []
    fut_norms = []
    ratios = []
    bad = 0
    total = 0

    with torch.no_grad():
        for bi, batch_pack in enumerate(make_seq_batches(examples, args.batch_size, device, shuffle=False)):
            if args.max_batches is not None and bi >= args.max_batches:
                break

            batch, *_ = batch_pack

            z = model.context_encoder(batch)

            delta = torch.full((z.shape[0], 1), args.delta_h, dtype=torch.float32, device=device)

            z_future = model.predictor(z, delta)

            v_current = torch.sum(z ** 2, dim=1)
            v_future = torch.sum(z_future ** 2, dim=1)

            delta_v = v_future - v_current

            for vc, vf, dv in zip(v_current.cpu(), v_future.cpu(), delta_v.cpu()):
                vc = safe_float(vc.item())
                vf = safe_float(vf.item())
                dv = safe_float(dv.item())

                total += 1

                if vc is None or vf is None or dv is None:
                    bad += 1
                    continue

                cur_norm = math.sqrt(max(vc, 0.0))
                fut_norm = math.sqrt(max(vf, 0.0))
                ratio = vf / (vc + 1e-8)

                cur_norms.append(cur_norm)
                fut_norms.append(fut_norm)
                delta_vs.append(dv)
                ratios.append(ratio)

                if vf > 1000 or ratio > 10:
                    bad += 1

    def stat(name, arr):
        arr = np.array(arr, dtype=float)
        if len(arr) == 0:
            print(f"{name}: no data")
            return
        print(
            f"{name}: "
            f"mean={arr.mean():.4f} "
            f"p50={np.percentile(arr, 50):.4f} "
            f"p90={np.percentile(arr, 90):.4f} "
            f"p99={np.percentile(arr, 99):.4f} "
            f"max={arr.max():.4f}"
        )

    print("\n=== Lyapunov-style latent stability check ===")
    stat("current ||z||", cur_norms)
    stat("future  ||z||", fut_norms)
    stat("delta V = V_future - V_current", delta_vs)
    stat("V_future / V_current", ratios)

    delta_vs_np = np.array(delta_vs, dtype=float)
    ratios_np = np.array(ratios, dtype=float)

    if len(delta_vs_np) > 0:
        pos_rate = float(np.mean(delta_vs_np > 0))
        big_growth_rate = float(np.mean(ratios_np > 2.0))
        print(f"\npositive delta_V rate: {pos_rate:.3f}")
        print(f"large growth ratio > 2 rate: {big_growth_rate:.3f}")
        print(f"bad / total: {bad} / {total}")

        if bad > 0 or big_growth_rate > 0.05:
            print("\n[warning] possible latent instability / explosion risk")
        elif pos_rate > 0.8:
            print("\n[caution] delta_V is often positive; model may amplify latent states")
        else:
            print("\n[ok] no obvious latent explosion detected")

    print("\nNote: this is an empirical Lyapunov diagnostic, not a formal proof.")


if __name__ == "__main__":
    main()
