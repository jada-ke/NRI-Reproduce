from __future__ import print_function
import numpy as np
import os

DATASETS = {
    "charged": {
        "train": [
            "loc_train_charged5.npy",
            "vel_train_charged5.npy",
        ],
        "eval": [
            "loc_valid_charged5.npy",
            "vel_valid_charged5.npy",
            "loc_test_charged5.npy",
            "vel_test_charged5.npy",
        ],
        "edges": [
            "edges_train_charged5.npy",
            "edges_valid_charged5.npy",
            "edges_test_charged5.npy",
        ],
    },
    "springs": {
        "train": [
            "loc_train_springs5.npy",
            "vel_train_springs5.npy",
        ],
        "eval": [
            "loc_valid_springs5.npy",
            "vel_valid_springs5.npy",
            "loc_test_springs5.npy",
            "vel_test_springs5.npy",
        ],
        "edges": [
            "edges_train_springs5.npy",
            "edges_valid_springs5.npy",
            "edges_test_springs5.npy",
        ],
    },
}


def apply_mcar_node_level(data, missing_rate, rng):
    """Masks entire nodes at each timestep independently of data values."""
    S, T, F, N = data.shape
    node_mask = rng.random_sample((S, T, N)) < missing_rate  # (S, T, N)
    mask = np.broadcast_to(node_mask[:, :, np.newaxis, :], data.shape).copy()
    data_out = data.copy().astype(float)
    data_out[mask] = np.nan
    return data_out, mask


def apply_mar_node_level(data, missing_rate, rng):
    """Probability of missingness depends on the magnitude of the observed reference signal."""
    S, T, F, N = data.shape
    data_out = data.copy().astype(float)
    node_mask = np.zeros((S, T, N), dtype=bool)

    for node_idx in range(N):
        ref_idx = (node_idx + 1) % N
        ref_signal = np.linalg.norm(data[:, :, :, ref_idx], axis=2)  # (S, T)
        ref_median = np.median(ref_signal)

        prob = np.where(ref_signal > ref_median,
                        missing_rate * 1.5,
                        missing_rate * 0.5)
        prob = np.clip(prob, 0, 1)
        node_mask[:, :, node_idx] = rng.random_sample(prob.shape) < prob

    mask = np.broadcast_to(node_mask[:, :, np.newaxis, :], data.shape).copy()
    data_out[mask] = np.nan
    return data_out, mask


def save_data(out_dir, filename, data, mask, suffix):
    name, ext = os.path.splitext(filename)
    np.save(os.path.join(out_dir, name + '_' + suffix + ext), data)
    np.save(os.path.join(out_dir, 'mask_' + name + '_' + suffix + ext), mask)


def generate_missing_datasets(dataset_name, base_input_dir, base_output_dir,
                               missing_rates, seed=42):
    """Generate MCAR and MAR masked datasets.

    Saved mask convention: True=missing, False=observed.
    Trajectory files have NaN at missing positions.
    """
    rng = np.random.RandomState(seed)
    files = DATASETS[dataset_name]

    for rate in missing_rates:
        rate_str = '{:02d}'.format(int(rate * 100))

        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        all_files = files["train"] + files["eval"]
        loc_files = [f for f in all_files if f.startswith("loc")]
        vel_files = [f for f in all_files if f.startswith("vel")]

        for loc_f, vel_f in zip(loc_files, vel_files):
            loc_data = np.load(os.path.join(base_input_dir, loc_f))
            vel_data = np.load(os.path.join(base_input_dir, vel_f))

            suffix_mcar = 'mcar' + rate_str
            suffix_mar  = 'mar'  + rate_str

            loc_mcar, mcar_mask = apply_mcar_node_level(loc_data, rate, rng)
            vel_mcar = vel_data.copy().astype(float)
            vel_mcar[mcar_mask] = np.nan
            save_data(base_output_dir, loc_f, loc_mcar, mcar_mask, suffix_mcar)
            save_data(base_output_dir, vel_f, vel_mcar, mcar_mask, suffix_mcar)

            loc_mar, mar_mask = apply_mar_node_level(loc_data, rate, rng)
            vel_mar = vel_data.copy().astype(float)
            vel_mar[mar_mask] = np.nan
            save_data(base_output_dir, loc_f, loc_mar, mar_mask, suffix_mar)
            save_data(base_output_dir, vel_f, vel_mar, mar_mask, suffix_mar)

            print('[{}] Processed {}/{} at rate {}'.format(
                dataset_name, loc_f, vel_f, rate))

        for edge_f in files["edges"]:
            edges = np.load(os.path.join(base_input_dir, edge_f))
            name, ext = os.path.splitext(edge_f)
            np.save(os.path.join(base_output_dir,
                                 name + '_mcar' + rate_str + ext), edges)
            np.save(os.path.join(base_output_dir,
                                 name + '_mar'  + rate_str + ext), edges)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='springs')
    parser.add_argument('--input-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='data')
    parser.add_argument('--rates', type=float, nargs='+', default=[0.3, 0.5])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generate_missing_datasets(
        dataset_name=args.dataset,
        base_input_dir=args.input_dir,
        base_output_dir=args.output_dir,
        missing_rates=args.rates,
        seed=args.seed,
    )
