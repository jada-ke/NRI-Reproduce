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
    node_mask = rng.random((*data.shape[:2], data.shape[3])) < missing_rate  # (S, T, N)
    mask = np.broadcast_to(node_mask[:, :, np.newaxis, :], data.shape).copy() # (S, T, F, N)
    data_out = data.copy().astype(float)
    data_out[mask] = np.nan
    return data_out, mask

def apply_mar_node_level(data, missing_rate, rng):
    """Probability of missingness depends on the magnitude of the observed reference signal."""
    data_out = data.copy().astype(float)
    num_nodes = data.shape[3]
    node_mask = np.zeros((*data.shape[:2], num_nodes), dtype=bool)

    for node_idx in range(num_nodes):
        ref_idx = (node_idx + 1) % num_nodes
        # Trigger based on distance from origin (Euclidean norm)
        ref_signal = np.linalg.norm(data[:, :, :, ref_idx], axis=2)  # (S, T)
        ref_median = np.median(ref_signal)

        # Higher probability of missingness for values above median
        prob = np.where(ref_signal > ref_median, missing_rate * 1.5, missing_rate * 0.5)
        prob = np.clip(prob, 0, 1)

        node_mask[:, :, node_idx] = rng.random(prob.shape) < prob

    mask = np.broadcast_to(node_mask[:, :, np.newaxis, :], data.shape).copy()
    data_out[mask] = np.nan
    return data_out, mask

def save_data(out_dir, filename, data, mask, suffix):
    name, ext = os.path.splitext(filename)

    new_filename = f"{name}_{suffix}{ext}"
    mask_filename = f"mask_{name}_{suffix}{ext}"

    np.save(os.path.join(out_dir, new_filename), data)
    np.save(os.path.join(out_dir, mask_filename), mask)

def generate_missing_datasets(dataset_name, base_input_dir, base_output_dir, missing_rates, seed=42):
    """Mask files:
    These are boolean arrays (True/False or 1/0) of the exact same shape as your trajectory files.
    True (or 1): Means "This data point is missing/masked."
    False (or 0): Means "This data point is original/valid."

    loc or vel files: corrupted files with NaN values
"""
    rng = np.random.default_rng(seed)
    files = DATASETS[dataset_name]

    for rate in missing_rates:
        rate_str = f"{int(rate * 100):02d}"
        os.makedirs(base_output_dir, exist_ok=True)

        # Combine all files into pairs (loc, vel)
        all_files = files["train"] + files["eval"]
        loc_files = [f for f in all_files if f.startswith("loc")]
        vel_files = [f for f in all_files if f.startswith("vel")]

        for loc_f, vel_f in zip(loc_files, vel_files):
            loc_data = np.load(os.path.join(base_input_dir, loc_f))
            vel_data = np.load(os.path.join(base_input_dir, vel_f))
            suffix_mcar = f"mcar{rate_str}"
            suffix_mar  = f"mar{rate_str}"

            # MCAR: Apply same mask to both features
            loc_mcar, mcar_mask = apply_mcar_node_level(loc_data, rate, rng)
            vel_mcar = vel_data.copy().astype(float)
            vel_mcar[mcar_mask] = np.nan
            save_data(base_output_dir, loc_f, loc_mcar, mcar_mask, suffix_mcar)
            save_data(base_output_dir, vel_f, vel_mcar, mcar_mask, suffix_mcar)

            # MAR: Apply same mask to both features
            loc_mar, mar_mask = apply_mar_node_level(loc_data, rate, rng)
            vel_mar = vel_data.copy().astype(float)
            vel_mar[mar_mask] = np.nan
            save_data(base_output_dir, loc_f, loc_mar, mar_mask, suffix_mar)
            save_data(base_output_dir, vel_f, vel_mar, mar_mask, suffix_mar)

            print(f"[{dataset_name}] Processed {loc_f}/{vel_f} at rate {rate}")

        for edge_f in files["edges"]:
            edges = np.load(os.path.join(base_input_dir, edge_f))

            name, ext = os.path.splitext(edge_f)

            np.save(os.path.join(base_output_dir, f"{name}_mcar{rate_str}{ext}"), edges)
            np.save(os.path.join(base_output_dir, f"{name}_mar{rate_str}{ext}"), edges)

if __name__ == "__main__":
    generate_missing_datasets(
        dataset_name="springs", 
        base_input_dir="data", 
        base_output_dir="data", 
        missing_rates=[0.1]
    )
