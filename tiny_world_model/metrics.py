# tiny_world_model/metrics.py
import torch
import numpy as np

def gaussian_nll(pred, target, sigma=0.05):
    """
    Simple Gaussian NLL with fixed sigma.
    pred, target: [B, T, 1]
    """
    var = sigma ** 2
    return 0.5 * ((target - pred) ** 2 / var + torch.log(torch.tensor(2 * np.pi * var)))


def compute_delta_nll(model_nll, baseline_nll):
    """
    ΔNLL = NLL_baseline - NLL_model (higher is better)
    """
    return baseline_nll - model_nll


def perm_p_value(observed_diff, perm_diffs, direction="higher", sided="one"):
    perm_diffs = np.asarray(perm_diffs)
    if sided == "one":
        if direction == "higher":
            count = np.sum(perm_diffs >= observed_diff)
        else:
            count = np.sum(perm_diffs <= observed_diff)
    else:
        obs = np.abs(observed_diff)
        perm_abs = np.abs(perm_diffs)
        count = np.sum(perm_abs >= obs)
    return (count + 1) / (len(perm_diffs) + 1)


def two_sample_permutation_test(struct_vals, null_vals, n_permutations=2000,
                                  direction="higher", sided="one"):
    """
    Two-sample permutation test for structured vs null comparison.

    Args:
        struct_vals: array of metric values from structured world
        null_vals: array of metric values from null world
        n_permutations: number of permutation samples
        direction: "higher" or "lower" for one-sided tests
        sided: "one" or "two"

    Returns:
        dict with observed_diff, p_value, null_distribution_mean, null_distribution_std
    """
    struct_vals = np.asarray(struct_vals)
    null_vals = np.asarray(null_vals)

    observed_diff = struct_vals.mean() - null_vals.mean()

    # Pool all values
    pooled = np.concatenate([struct_vals, null_vals])
    n_struct = len(struct_vals)
    n_total = len(pooled)

    perm_diffs = []
    for _ in range(n_permutations):
        perm_idx = np.random.permutation(n_total)
        perm_struct = pooled[perm_idx[:n_struct]]
        perm_null = pooled[perm_idx[n_struct:]]
        perm_diffs.append(perm_struct.mean() - perm_null.mean())

    p_value = perm_p_value(observed_diff, perm_diffs, direction=direction, sided=sided)

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "null_distribution_mean": float(np.mean(perm_diffs)),
        "null_distribution_std": float(np.std(perm_diffs)),
        "n_permutations": n_permutations
    }
