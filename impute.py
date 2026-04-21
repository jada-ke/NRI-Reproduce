"""
Graph-Guided Iterative Imputation (GGII) for NRI.

Stage 2 of the two-stage GGII pipeline:
  1. Train NRI with mean imputation on a missing-data dataset (train.py).
  2. Run this script to produce a graph-guided imputed dataset.
  3. Retrain NRI on the improved dataset (train.py, new suffix).

Why this outperforms mean imputation at high missingness:
  - Mean imputation places a missing atom at its temporal average position,
    ignoring the current phase of its oscillation.
  - The NRI decoder knows spring physics AND the inferred graph structure, so
    it can propagate the last observed state forward in time, producing a
    physically plausible estimate of the missing position.
  - At 10% missing the gap is small and the difference is minor; at 50%+
    missing the decoder prediction can be far better than the temporal mean.

Usage (on the Mila cluster):
  python impute.py \
    --suffix _springs5_mar50 \
    --load-folder $SCRATCH/NRI/logs/exp<timestamp>/ \
    --out-suffix _springs5_mar50_ggii

Then retrain:
  python train.py \
    --suffix _springs5_mar50_ggii \
    --skip-first \
    --save-folder $SCRATCH/NRI/logs
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.enabled = False  # avoid CUDNN_STATUS_NOT_SUPPORTED in eval BatchNorm
import torch.nn.functional as F
from torch.autograd import Variable

from utils import mean_impute, encode_onehot, my_softmax
from modules import MLPEncoder, MLPDecoder, MaskedPairwiseEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--suffix', type=str, default='_springs5_mar10',
                    help='Suffix of the source dataset (must have mask files).')
parser.add_argument('--out-suffix', type=str, default='',
                    help='Suffix for the imputed output dataset. '
                         'Defaults to <suffix>_ggii.')
parser.add_argument('--load-folder', type=str, required=True,
                    help='Folder containing encoder.pt / decoder.pt '
                         'trained with mean imputation on --suffix data.')
parser.add_argument('--num-atoms', type=int, default=5)
parser.add_argument('--dims', type=int, default=4,
                    help='Feature dims (must match the trained model).')
parser.add_argument('--timesteps', type=int, default=49)
parser.add_argument('--encoder-hidden', type=int, default=256)
parser.add_argument('--decoder-hidden', type=int, default=256)
parser.add_argument('--edge-types', type=int, default=2)
parser.add_argument('--encoder-dropout', type=float, default=0.0)
parser.add_argument('--decoder-dropout', type=float, default=0.0)
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Must match how the loaded model was trained.')
parser.add_argument('--factor', action='store_true', default=True)
parser.add_argument('--use-mask', action='store_true', default=False,
                    help='Load MaskedPairwiseEncoder (must match how the model was trained).')
parser.add_argument('--batch-size', type=int, default=256,
                    help='Batch size for inference (larger is faster).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not args.out_suffix:
    args.out_suffix = args.suffix + '_ggii'

print("Source suffix :", args.suffix)
print("Output suffix :", args.out_suffix)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_raw(split):
    """Load raw loc/vel (may contain NaN) and masks for one split."""
    loc = np.load('data/loc_' + split + args.suffix + '.npy')
    vel = np.load('data/vel_' + split + args.suffix + '.npy')
    # Masks: 1 = observed, 0 = missing.  May not exist (clean dataset).
    mloc_path = 'data/mask_loc_' + split + args.suffix + '.npy'
    mvel_path = 'data/mask_vel_' + split + args.suffix + '.npy'
    ref = loc.shape  # [sims, T, 2, atoms]
    if os.path.exists(mloc_path):
        mloc = 1.0 - np.load(mloc_path).astype(np.float32)  # flip: True=missing -> 0=missing
        if mloc.ndim == 3:
            mloc = np.stack([mloc, mloc], axis=2)
    else:
        mloc = np.ones(ref, dtype=np.float32)
    if os.path.exists(mvel_path):
        mvel = 1.0 - np.load(mvel_path).astype(np.float32)
        if mvel.ndim == 3:
            mvel = np.stack([mvel, mvel], axis=2)
    else:
        mvel = np.ones(ref, dtype=np.float32)
    return loc, vel, mloc, mvel


def normalize(loc, vel, loc_min, loc_max, vel_min, vel_max):
    loc = (loc - loc_min) * 2.0 / (loc_max - loc_min) - 1.0
    vel = (vel - vel_min) * 2.0 / (vel_max - vel_min) - 1.0
    return loc, vel


def unnormalize(loc, vel, loc_min, loc_max, vel_min, vel_max):
    loc = 0.5 * (loc + 1.0) * (loc_max - loc_min) + loc_min
    vel = 0.5 * (vel + 1.0) * (vel_max - vel_min) + vel_min
    return loc, vel


def to_feat(loc_norm, vel_norm):
    """Stack into [sims, atoms, T, 4] feature tensor."""
    # loc_norm / vel_norm: [sims, T, 2, atoms] -> [sims, atoms, T, 2]
    loc = np.transpose(loc_norm, [0, 3, 1, 2])
    vel = np.transpose(vel_norm, [0, 3, 1, 2])
    return np.concatenate([loc, vel], axis=3)  # [sims, atoms, T, 4]


# ---------------------------------------------------------------------------
# Build rel_rec / rel_send
# ---------------------------------------------------------------------------
N = args.num_atoms
off_diag = np.ones([N, N]) - np.eye(N)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)
if args.cuda:
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)

# ---------------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------------
if args.use_mask:
    encoder = MaskedPairwiseEncoder(args.dims, args.encoder_hidden,
                                    args.edge_types, args.encoder_dropout,
                                    args.factor)
else:
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types, args.encoder_dropout, args.factor)
decoder = MLPDecoder(n_in_node=args.dims,
                     edge_types=args.edge_types,
                     msg_hid=args.decoder_hidden,
                     msg_out=args.decoder_hidden,
                     n_hid=args.decoder_hidden,
                     do_prob=args.decoder_dropout,
                     skip_first=args.skip_first)

encoder.load_state_dict(torch.load(os.path.join(args.load_folder, 'encoder.pt')))
decoder.load_state_dict(torch.load(os.path.join(args.load_folder, 'decoder.pt')))
encoder.eval()
decoder.eval()
if args.cuda:
    encoder.cuda()
    decoder.cuda()

print("Model loaded from", args.load_folder)

# ---------------------------------------------------------------------------
# Compute normalization constants from training set (mean-imputed)
# ---------------------------------------------------------------------------
loc_tr_raw, vel_tr_raw, mloc_tr, mvel_tr = load_raw('train')
loc_tr_mi = mean_impute(loc_tr_raw)
vel_tr_mi = mean_impute(vel_tr_raw)
loc_max = loc_tr_mi.max()
loc_min = loc_tr_mi.min()
vel_max = vel_tr_mi.max()
vel_min = vel_tr_mi.min()
print("Normalization  loc:[{:.3f},{:.3f}]  vel:[{:.3f},{:.3f}]".format(
    loc_min, loc_max, vel_min, vel_max))


# ---------------------------------------------------------------------------
# Core: graph-guided sequential imputation
# ---------------------------------------------------------------------------

def graph_guided_impute(loc_raw, vel_raw, mask_loc, mask_vel,
                        loc_min, loc_max, vel_min, vel_max,
                        split_name=''):
    """
    For each simulation:
      1. Mean-impute and normalize.
      2. Run encoder -> soft edge-type beliefs.
      3. Sequentially roll out the decoder; at each timestep replace the
         NaN entries with the decoder's prediction.
      4. Return un-normalized imputed arrays (same shape as inputs).

    mask_loc / mask_vel: [sims, T, 2, atoms], 1=observed 0=missing.
    """
    S, T, D2, A = loc_raw.shape  # D2=2 for loc, 2 for vel -> dims=4 total
    print("  Imputing {} {} sims ...".format(S, split_name))

    loc_mi = mean_impute(loc_raw)   # [sims, T, 2, atoms]
    vel_mi = mean_impute(vel_raw)

    loc_norm, vel_norm = normalize(loc_mi, vel_mi, loc_min, loc_max,
                                   vel_min, vel_max)
    feat_mi = to_feat(loc_norm, vel_norm)   # [sims, atoms, T, 4]

    # Combined observation mask: [sims, atoms, T, 4]
    mask_loc_t = np.transpose(mask_loc, [0, 3, 1, 2])  # [sims, atoms, T, 2]
    mask_vel_t = np.transpose(mask_vel, [0, 3, 1, 2])
    mask_full = np.concatenate([mask_loc_t, mask_vel_t], axis=3)  # [sims,A,T,4]

    loc_out = loc_norm.copy()   # will be modified where mask=0
    vel_out = vel_norm.copy()

    B = args.batch_size
    for start in range(0, S, B):
        end = min(start + B, S)
        feat_b = feat_mi[start:end]                    # [B, A, T, 4]
        if args.use_mask:
            mask_b = mask_full[start:end]              # [B, A, T, 4]
            enc_in = torch.FloatTensor(
                np.concatenate([feat_b, mask_b], axis=3))  # [B, A, T, 8]
        else:
            enc_in = torch.FloatTensor(
                feat_b.reshape(feat_b.shape[0], feat_b.shape[1], -1))  # [B, A, T*4]
        if args.cuda:
            enc_in = enc_in.cuda()
        enc_in = Variable(enc_in, volatile=True)

        # Step 1: encoder -> edge beliefs
        logits = encoder(enc_in, rel_rec, rel_send)   # [B, E, edge_types]
        edges = my_softmax(logits, -1)                # soft beliefs

        # Step 2: sequential rollout - replace missing steps with predictions
        # Work in normalized space on CPU numpy for simplicity
        feat_np = feat_mi[start:end].copy()           # [b, A, T, 4]
        mask_np = mask_full[start:end]                # [b, A, T, 4]
        edges_np = edges.data.cpu().numpy()

        for t in range(1, T):
            missing_t = mask_np[:, :, t, :]  # [b, A, 4], 0=missing
            if missing_t.min() == 1:
                continue  # all atoms observed at this step, nothing to impute

            # Build decoder input: [b, 1, A, 4] (single_step_forward expects time as dim 1)
            prev = torch.FloatTensor(feat_np[:, :, t-1, :][:, np.newaxis, :, :])  # [b, 1, A, 4]
            e_var = torch.FloatTensor(edges_np).unsqueeze(1)  # [b, 1, E, edge_types]
            if args.cuda:
                prev = prev.cuda()
                e_var = e_var.cuda()
            prev = Variable(prev, volatile=True)
            e_var = Variable(e_var, volatile=True)

            pred = decoder.single_step_forward(prev, rel_rec, rel_send, e_var)
            pred_np = pred.data.cpu().numpy()[:, 0, :, :]  # [b, A, 4]

            # Replace missing entries with decoder prediction
            # miss: [b, A] bool - True where this atom is missing at step t
            miss = (mask_np[:, :, t, :].min(axis=2) == 0)  # [b, A]
            for bi in range(miss.shape[0]):
                for ai in range(miss.shape[1]):
                    if miss[bi, ai]:
                        feat_np[bi, ai, t, :] = pred_np[bi, ai, :]

        # Unpack back to loc/vel in [sims, T, 2, atoms]
        # feat_np: [b, A, T, 4] -> loc [b, A, T, 2], vel [b, A, T, 2]
        loc_imp = feat_np[:, :, :, :2]   # [b, A, T, 2]
        vel_imp = feat_np[:, :, :, 2:]   # [b, A, T, 2]
        loc_out[start:end] = np.transpose(loc_imp, [0, 2, 3, 1])  # [b, T, 2, A]
        vel_out[start:end] = np.transpose(vel_imp, [0, 2, 3, 1])

    return loc_out, vel_out   # still normalized


# ---------------------------------------------------------------------------
# Impute all splits and save
# ---------------------------------------------------------------------------

splits = [
    ('train', loc_tr_raw, vel_tr_raw, mloc_tr, mvel_tr),
]
for split_name in ('valid', 'test'):
    l, v, ml, mv = load_raw(split_name)
    splits.append((split_name, l, v, ml, mv))

for split_name, loc_raw, vel_raw, mloc, mvel in splits:
    loc_imp_norm, vel_imp_norm = graph_guided_impute(
        loc_raw, vel_raw, mloc, mvel,
        loc_min, loc_max, vel_min, vel_max,
        split_name=split_name)

    # Un-normalize back to original scale before saving
    loc_imp, vel_imp = unnormalize(loc_imp_norm, vel_imp_norm,
                                   loc_min, loc_max, vel_min, vel_max)

    # Save: load_data expects the same format as the original datasets.
    # We save the imputed arrays (no NaN) and copy the edge files.
    out_loc = 'data/loc_' + split_name + args.out_suffix + '.npy'
    out_vel = 'data/vel_' + split_name + args.out_suffix + '.npy'
    np.save(out_loc, loc_imp.astype(np.float32))
    np.save(out_vel, vel_imp.astype(np.float32))
    print("Saved", out_loc)

    # Copy edge file (graph structure is unchanged)
    edges = np.load('data/edges_' + split_name + args.suffix + '.npy')
    np.save('data/edges_' + split_name + args.out_suffix + '.npy', edges)

print("Done. Retrain with:")
print("  python train.py --suffix {} --skip-first --save-folder $SCRATCH/NRI/logs".format(
    args.out_suffix))
