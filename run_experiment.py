# run_experiment.py
import os
import json
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime

from tiny_world_model.utils import set_all_seeds
from tiny_world_model.worlds.drift import DriftWorld, DriftNullWorld, DriftDecoySineWorld
from tiny_world_model.worlds.oscillator import OscillatorWorld, OscillatorNullWorld, OscillatorNonstatWorld
from tiny_world_model.worlds.regime import RegimeMarkovWorld, RegimeNullWorld
from tiny_world_model.models.rnn import TinyRNN, TinyGRU, TinyLSTM
from tiny_world_model.time_cond import apply_time_cond
from tiny_world_model.metrics import gaussian_nll, compute_delta_nll, perm_p_value
from tiny_world_model.probes import train_probe, probe_with_controls


def scramble_model_core(model):
    """
    Randomly permute the entries of each recurrent weight tensor.
    This destroys the learned dynamics while preserving the geometry/manifold.
    """
    import torch

    with torch.no_grad():
        # Find the recurrent core (rnn, gru, or lstm)
        core = None
        if hasattr(model, "rnn"):
            core = model.rnn
        elif hasattr(model, "gru"):
            core = model.gru
        elif hasattr(model, "lstm"):
            core = model.lstm

        if core is None:
            print("[WARN] No recurrent core found to scramble")
            return

        # Scramble each weight tensor
        for name, p in core.named_parameters():
            # Only scramble weight matrices (2D or higher)
            if p.dim() >= 2:
                flat = p.view(-1)
                perm = torch.randperm(flat.numel(), device=flat.device)
                flat.copy_(flat[perm])
                print(f"  → Scrambled {name} ({p.shape})")


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("\n[DEBUG] Starting experiment...")

    # Setup output directory for this run
    output_dir = Path.cwd()  # Hydra sets CWD to outputs/<date>/<time>
    results_file = output_dir / "results.json"

    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    # Build world
    if cfg.world.type == "drift":
        world = DriftWorld(cfg.world, device=device)
    elif cfg.world.type == "drift_null":
        world = DriftNullWorld(cfg.world, device=device)
    elif cfg.world.type == "drift_decoy_sine":
        world = DriftDecoySineWorld(cfg.world, device=device)
    elif cfg.world.type == "oscillator":
        world = OscillatorWorld(cfg.world, device=device)
    elif cfg.world.type == "oscillator_null":
        world = OscillatorNullWorld(cfg.world, device=device)
    elif cfg.world.type == "oscillator_nonstat":
        world = OscillatorNonstatWorld(cfg.world, device=device)
    elif cfg.world.type == "regime_markov":
        world = RegimeMarkovWorld(cfg.world, device=device)
    elif cfg.world.type == "regime_null":
        world = RegimeNullWorld(cfg.world, device=device)
    else:
        raise NotImplementedError(f"World type {cfg.world.type} not implemented in this script.")

    # Container for per-seed metrics
    delta_nll_struct_seeds = []
    per_seed_results = []

    print(f"\n[DEBUG] Starting {cfg.experiment.n_seeds} seeds from master seed {cfg.random.master_seed}")

    master = cfg.random.master_seed
    for i in range(cfg.experiment.n_seeds):
        seed = master + i
        print(f"\n=== Seed {seed} ===")
        set_all_seeds(seed)

        seed_result = {
            "seed": seed,
            "seed_index": i
        }

        # Build model
        input_dim = cfg.world.input_dim
        hidden_dim = cfg.model.hidden_dim

        if cfg.model.type == "rnn_tanh":
            model = TinyRNN(input_dim=input_dim, hidden_dim=hidden_dim, init_cfg=cfg.model.init).to(device)
        elif cfg.model.type == "gru":
            model = TinyGRU(input_dim=input_dim, hidden_dim=hidden_dim, init_cfg=cfg.model.init).to(device)
        elif cfg.model.type == "lstm":
            model = TinyLSTM(input_dim=input_dim, hidden_dim=hidden_dim, init_cfg=cfg.model.init).to(device)
        else:
            raise NotImplementedError(f"Unknown model type: {cfg.model.type}")

        # Set up optimizer: full model vs readout-only
        readout_only = getattr(cfg.train, "readout_only", False)
        if readout_only:
            # Freeze recurrent core (RNN/GRU/LSTM)
            core = getattr(model, "rnn", None) or getattr(model, "gru", None) or getattr(model, "lstm", None)
            if core is not None:
                for p in core.parameters():
                    p.requires_grad_(False)
                # Only train the head
                opt = torch.optim.Adam(model.head.parameters(), lr=cfg.model.optimizer.lr)
                print(f"[seed {seed}] READOUT-ONLY mode: frozen {cfg.model.type} core, training head only.")
            else:
                raise RuntimeError(f"Could not find recurrent core in model type {cfg.model.type}")
        else:
            opt = torch.optim.Adam(model.parameters(), lr=cfg.model.optimizer.lr)
            print(f"[seed {seed}] FULL model training: {cfg.model.type} + head.")

        train_loader = world.train_dataloader(cfg.train.batch_size)
        eval_loader  = world.eval_dataloader(cfg.train.batch_size)

        # ---- TRAINING ----
        step = 0

        if cfg.train.max_steps <= 0:
            print(f"[seed {seed}] max_steps <= 0 → skipping training (random-weights control)")
        else:
            for epoch in range(9999):
                for x_in, y, z_lat in train_loader:
                    x_in, y = x_in.to(device), y.to(device)

                    # add time conditioning (regime worlds already include time)
                    if cfg.world.type.startswith("regime"):
                        x_tc = x_in
                    else:
                        x_tc = apply_time_cond(x_in, cfg.time_cond, world.T)

                    pred, h_seq = model(x_tc)
                    # next-step prediction loss
                    loss = gaussian_nll(pred[:, :-1], y[:, 1:]).mean()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    step += 1
                    if step % cfg.train.log_interval == 0:
                        print(f"[seed {seed}] step={step} loss={loss.item():.4f}")
                    if step >= cfg.train.max_steps:
                        break
                if step >= cfg.train.max_steps:
                    break

        # ---- SCRAMBLE core if requested (after training, before eval) ----
        scramble_core = getattr(cfg.train, "scramble_core", False)
        if scramble_core:
            print(f"[seed {seed}] SCRAMBLING recurrent core before evaluation.")
            scramble_model_core(model)

        # ---- EVAL on structured world ----
        model.eval()
        with torch.no_grad():
            total_nll_model = 0.0
            total_nll_base  = 0.0
            count = 0
            all_H = []
            all_X_lat = []

            for x_in, y, z_lat in eval_loader:
                x_in = x_in.to(device)
                y    = y.to(device)
                z_lat= z_lat.to(device)

                # add time conditioning (regime worlds already include time)
                if cfg.world.type.startswith("regime"):
                    x_tc = x_in
                else:
                    x_tc = apply_time_cond(x_in, cfg.time_cond, world.T)

                pred, h_seq = model(x_tc)

                nll_model = gaussian_nll(pred[:, :-1], y[:, 1:]).sum().item()

                # Smart trivial baseline: best of {persistence, global mean}
                y_true = y[:, 1:]          # [B, T-1, 1]
                y_prev = y[:, :-1]         # [B, T-1, 1]

                # Persistence predictor
                base_persist = y_prev

                # Global mean predictor (scalar from training data)
                baseline_val = world.baseline_mean
                base_mean = torch.full_like(y_true, baseline_val)

                # Compute MSE of each (as a proxy for NLL choice)
                mse_persist = torch.mean((y_true - base_persist) ** 2).item()
                mse_mean    = torch.mean((y_true - base_mean) ** 2).item()

                # Pick the better trivial strategy
                if mse_persist < mse_mean:
                    base_pred = base_persist
                else:
                    base_pred = base_mean

                nll_base = gaussian_nll(base_pred, y_true).sum().item()

                total_nll_model += nll_model
                total_nll_base  += nll_base
                count += (y.shape[0] * (y.shape[1] - 1))

                # store hidden + latent for probe
                all_H.append(h_seq[:, 1:].reshape(-1, h_seq.shape[-1]))

                # Latent targets depend on world type
                if cfg.world.type.startswith("oscillator"):
                    # For all oscillator variants (struct, null, nonstat), latent_dim=2 (x, y).
                    # We probe only x_t (first coordinate) as scalar target.
                    z_target = z_lat[:, 1:, 0:1]    # [B, T-1, 1]
                else:
                    # E1 drift and other worlds
                    z_target = z_lat[:, 1:]         # [B, T-1, 1]

                all_X_lat.append(z_target.reshape(-1, 1))

            avg_nll_model = total_nll_model / count
            avg_nll_base  = total_nll_base  / count
            delta_nll = compute_delta_nll(avg_nll_model, avg_nll_base)

            print(f"[seed {seed}] avg_nll_model={avg_nll_model:.4f} "
                  f"avg_nll_base={avg_nll_base:.4f} ΔNLL={delta_nll:.4f}")

            delta_nll_struct_seeds.append(delta_nll)

            # Store metrics for this seed
            seed_result["avg_nll_model"] = float(avg_nll_model)
            seed_result["avg_nll_baseline"] = float(avg_nll_base)
            seed_result["delta_nll"] = float(delta_nll)
            seed_result["n_eval_samples"] = int(count)

            H_all = torch.cat(all_H, dim=0)
            X_all = torch.cat(all_X_lat, dim=0)

        # ---- Probe on latent x_t with controls ----
        probe_results = probe_with_controls(H_all, X_all)
        print(f"[seed {seed}] probe R² trained={probe_results['trained']['r2']:.4f} "
              f"random={probe_results['random']['r2']:.4f} "
              f"permuted={probe_results['permuted']['r2']:.4f}")

        seed_result["probe_r2_trained"] = float(probe_results['trained']['r2'])
        seed_result["probe_mse_trained"] = float(probe_results['trained']['mse'])
        seed_result["probe_r2_random"] = float(probe_results['random']['r2'])
        seed_result["probe_mse_random"] = float(probe_results['random']['mse'])
        seed_result["probe_r2_permuted"] = float(probe_results['permuted']['r2'])
        seed_result["probe_mse_permuted"] = float(probe_results['permuted']['mse'])

        per_seed_results.append(seed_result)

    # ---- Permutation Test on ΔNLL vs 0 ----
    # Here we treat "null" as zero-improvement baseline.
    struct_vals = np.array(delta_nll_struct_seeds)
    # synthetic null distribution via sign-flip
    pooled = struct_vals.copy()
    n = len(pooled)
    observed_diff = struct_vals.mean()  # vs 0

    perm_diffs = []
    for _ in range(cfg.stats.n_permutations):
        signs = np.random.choice([-1, 1], size=n)
        perm_diffs.append((pooled * signs).mean())

    p_perm = perm_p_value(observed_diff, perm_diffs,
                          direction=cfg.stats.metrics.delta_nll.direction,
                          sided=cfg.stats.metrics.delta_nll.sided)
    print(f"\n[Permutation] ΔNLL mean={observed_diff:.4f}, p_perm={p_perm:.5f}")

    # ---- Write results to JSON ----
    readout_only = getattr(cfg.train, "readout_only", False)
    scramble_core = getattr(cfg.train, "scramble_core", False)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "world_id": cfg.world.id,
            "world_type": cfg.world.type,
            "model_id": cfg.model.id,
            "model_type": cfg.model.type,
            "time_cond_type": cfg.time_cond.type,
            "readout_only": bool(readout_only),
            "scramble_core": bool(scramble_core),
            "n_seeds": cfg.experiment.n_seeds,
            "master_seed": cfg.random.master_seed,
            "device": str(device),
            "output_dir": str(output_dir)
        },
        "hyperparameters": {
            "hidden_dim": cfg.model.hidden_dim,
            "learning_rate": cfg.model.optimizer.lr,
            "batch_size": cfg.train.batch_size,
            "max_steps": cfg.train.max_steps,
            "n_train_seqs": cfg.world.data_gen.n_train_seqs,
            "n_eval_seqs": cfg.world.data_gen.n_eval_seqs,
            "sequence_length": world.T
        },
        "per_seed_results": per_seed_results,
        "aggregated_metrics": {
            "delta_nll": {
                "mean": float(np.mean(delta_nll_struct_seeds)),
                "std": float(np.std(delta_nll_struct_seeds)),
                "min": float(np.min(delta_nll_struct_seeds)),
                "max": float(np.max(delta_nll_struct_seeds)),
                "values": [float(x) for x in delta_nll_struct_seeds]
            },
            "probe_r2_trained": {
                "mean": float(np.mean([r["probe_r2_trained"] for r in per_seed_results])),
                "std": float(np.std([r["probe_r2_trained"] for r in per_seed_results])),
                "values": [float(r["probe_r2_trained"]) for r in per_seed_results]
            },
            "probe_r2_random": {
                "mean": float(np.mean([r["probe_r2_random"] for r in per_seed_results])),
                "std": float(np.std([r["probe_r2_random"] for r in per_seed_results])),
                "values": [float(r["probe_r2_random"]) for r in per_seed_results]
            },
            "probe_r2_permuted": {
                "mean": float(np.mean([r["probe_r2_permuted"] for r in per_seed_results])),
                "std": float(np.std([r["probe_r2_permuted"] for r in per_seed_results])),
                "values": [float(r["probe_r2_permuted"]) for r in per_seed_results]
            },
            "probe_emergence_margin": {
                "description": "trained R² minus max(random, permuted) R²",
                "mean": float(np.mean([
                    r["probe_r2_trained"] - max(r["probe_r2_random"], r["probe_r2_permuted"])
                    for r in per_seed_results
                ])),
                "values": [
                    float(r["probe_r2_trained"] - max(r["probe_r2_random"], r["probe_r2_permuted"]))
                    for r in per_seed_results
                ]
            }
        },
        "statistical_tests": {
            "permutation_test": {
                "metric": "delta_nll",
                "observed_mean": float(observed_diff),
                "p_value": float(p_perm),
                "n_permutations": cfg.stats.n_permutations,
                "direction": cfg.stats.metrics.delta_nll.direction,
                "sided": cfg.stats.metrics.delta_nll.sided,
                "null_distribution_mean": float(np.mean(perm_diffs)),
                "null_distribution_std": float(np.std(perm_diffs))
            }
        }
    }

    # Write full results to Hydra output dir (for debugging/inspection)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[RESULTS] Full results saved to: {results_file}")

    # ---- Also save to experiments/results for persistent archive ----
    # Build unique filename based on run parameters
    tag = f"world={cfg.world.id}_model={cfg.model.id}_hidden={cfg.model.hidden_dim}_ro={int(readout_only)}_scr={int(scramble_core)}"

    # Get project root (go up from wherever Hydra put us)
    project_root = Path(__file__).parent
    results_dir = project_root / "experiments" / "results"
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Save per-run JSON with unique name
    per_run_file = runs_dir / f"{tag}.json"
    with open(per_run_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[RESULTS] Per-run JSON saved to: {per_run_file}")

    # Validity check: detect exploded or invalid runs
    # Heuristics:
    # 1. NaN or Inf in delta_nll_mean
    # 2. Extremely negative delta_nll (< -500) suggests explosion
    # 3. For non-scrambled runs, delta_nll < -100 is suspicious
    is_valid = True
    max_reasonable_negative_dnll = -500.0 if scramble_core else -100.0

    if np.isnan(observed_diff) or np.isinf(observed_diff):
        is_valid = False
        print(f"[VALIDITY] Run marked invalid: delta_nll is NaN or Inf")
    elif observed_diff < max_reasonable_negative_dnll:
        is_valid = False
        print(f"[VALIDITY] Run marked invalid: delta_nll={observed_diff:.2f} < {max_reasonable_negative_dnll}")
    elif any(np.isnan(delta_nll_struct_seeds)) or any(np.isinf(delta_nll_struct_seeds)):
        is_valid = False
        print(f"[VALIDITY] Run marked invalid: NaN or Inf in per-seed delta_nll")
    else:
        print(f"[VALIDITY] Run marked valid")

    # Build compact run summary for append-only log
    run_summary = {
        "timestamp": results["metadata"]["timestamp"],
        "world_id": cfg.world.id,
        "world_type": cfg.world.type,
        "model_id": cfg.model.id,
        "model_type": cfg.model.type,
        "hidden_dim": cfg.model.hidden_dim,
        "readout_only": bool(readout_only),
        "scramble_core": bool(scramble_core),
        "n_seeds": cfg.experiment.n_seeds,
        "delta_nll_mean": float(observed_diff),
        "delta_nll_std": float(np.std(delta_nll_struct_seeds)),
        "delta_nll_p_perm": float(p_perm),
        "probe_r2_trained_mean": results["aggregated_metrics"]["probe_r2_trained"]["mean"],
        "probe_r2_random_mean": results["aggregated_metrics"]["probe_r2_random"]["mean"],
        "probe_r2_permuted_mean": results["aggregated_metrics"]["probe_r2_permuted"]["mean"],
        "probe_emergence_margin_mean": results["aggregated_metrics"]["probe_emergence_margin"]["mean"],
        "valid": is_valid,
    }

    # Append to summary.jsonl (one line per run)
    summary_path = results_dir / "summary.jsonl"
    with open(summary_path, 'a') as f:
        f.write(json.dumps(run_summary) + "\n")
    print(f"[RESULTS] Appended to summary log: {summary_path}")


if __name__ == "__main__":
    main()
