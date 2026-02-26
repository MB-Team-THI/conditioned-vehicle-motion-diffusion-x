import os
import random
import numpy as np
import pandas as pd

from improved_diffusion.vmm import vmm
from scripts.gmm_and_agg import gmm_clustering

# -------------------
# Config
# -------------------
MAX_ACC   = 4.0
MAX_dPSI  = 0.7

MODE      = "posthoc"        # "iter" or "posthoc"
NUM_SAMPLES_PER_SCENARIO = 9
K_MAX     = 3

# data
DDIM_STEPS = 10
CFG     = '1.0'
GUIDANCE_RESCALE = False

FOLDER  = "cl_90"                   # model folder name
DIR_    = "../ckpts/"               # model directory
if GUIDANCE_RESCALE:
    PICKLE  = os.path.join(DIR_, FOLDER, f"resultsNEWk/results_{DDIM_STEPS}_gencfg={CFG}r.pkl")
else:
    PICKLE  = os.path.join(DIR_, FOLDER, f"resultsNEWk/results_{DDIM_STEPS}_gencfg={CFG}.pkl")

# -------------------
# Metrics
# -------------------
def hit(traj_xy, gt_xy, delta=1.75):
    """Binary hit if final lateral error under delta."""
    traj_xy = np.asarray(traj_xy); gt_xy = np.asarray(gt_xy)
    return 1 if abs(traj_xy[-1, 1] - gt_xy[-1, 1]) < delta else 0

def mse_xy(traj_xy, gt_xy, pos=-1):
    """Mean squared error in (x,y) up to pos (exclusive if negative like -1 means full)."""
    traj_xy = np.asarray(traj_xy); gt_xy = np.asarray(gt_xy)
    end = None if pos < 0 else pos
    diff = traj_xy[:end] - gt_xy[:end]
    return np.mean(np.sum(diff * diff, axis=-1))

def ade_xy(traj_xy, gt_xy):
    """Average displacement error for 2D trajectories, supports (T,2) and (B,T,2)."""
    traj_xy = np.asarray(traj_xy); gt_xy = np.asarray(gt_xy)
    return np.mean(np.linalg.norm(traj_xy - gt_xy, axis=-1))

def fde_xy(traj_xy, gt_xy):
    """Final displacement error for 2D trajectories, supports (T,2) and (B,T,2)."""
    traj_xy = np.asarray(traj_xy); gt_xy = np.asarray(gt_xy)
    if traj_xy.ndim == 2:
        return np.linalg.norm(traj_xy[-1] - gt_xy[-1])
    else:
        return np.mean(np.linalg.norm(traj_xy[:, -1] - gt_xy[:, -1], axis=-1))

def minADE_k(gt_xy, trajectories_xy, labels, n_clusters, K=4):
    """Cluster-wise mean-trajectory ADE; take best over top-K clusters."""
    K = min(K, n_clusters)
    best = np.inf
    for k in range(K):
        mask = labels == k
        mean_traj = np.mean(trajectories_xy[mask], axis=0)  # (T,2)
        best = min(best, ade_xy(mean_traj, gt_xy))
    return best

def minFDE_k(gt_xy, trajectories_xy, labels, n_clusters, K=4):
    K = min(K, n_clusters)
    best = np.inf
    for k in range(K):
        mean_traj = np.mean(trajectories_xy[labels == k], axis=0)
        best = min(best, fde_xy(mean_traj, gt_xy))
    return best

def ade_most_likely(gt_xy, trajectories_xy, labels, cluster_probs, K):
    """
    ADE using the *most likely* cluster according to cluster_probs.
    trajectories_xy: (N, T, 2)
    labels: (N,)
    cluster_probs: (n_clusters,)
    """
    most_likely_cluster = np.argmax(cluster_probs)
    mask = labels == most_likely_cluster
    mean_traj = trajectories_xy[mask].mean(axis=0)  # (T, 2)
    return ade_xy(mean_traj, gt_xy)#, mean_traj

def fde_most_likely(gt_xy, trajectories_xy, labels, cluster_probs, K):
    """
    FDE using the *most likely* cluster according to cluster_probs.
    trajectories_xy: (N, T, 2)
    labels: (N,)
    cluster_probs: (n_clusters,)
    """
    most_likely_cluster = np.argmax(cluster_probs)
    mask = labels == most_likely_cluster
    mean_traj = trajectories_xy[mask].mean(axis=0)  # (T, 2)
    return fde_xy(mean_traj, gt_xy)#, mean_traj

# -------------------
# Accumulator to avoid "mean of means" bias
# -------------------
class MetricAccumulator:
    """
    Accumulate sums and counts so the final mean is correct even with
    different scenario lengths or skipped items.
    """
    def __init__(self):
        self.tot_ade_real = 0.0; self.n_ade_real = 0
        self.tot_fde_real = 0.0; self.n_fde_real = 0
        self.tot_mse_real = 0.0; self.n_mse_real = 0

        self.tot_ade_mean = 0.0; self.n_ade_mean = 0
        self.tot_fde_mean = 0.0; self.n_fde_mean = 0

        self.ml_ade_list = []
        self.ml_fde_list =[]

        self.min_ade_list = []
        self.min_fde_list = []
        self.hit_list     = []

        # optional per-axis ADE (MAE in x/y)
        self.tot_mae_x = 0.0; self.n_mae_x = 0
        self.tot_mae_y = 0.0; self.n_mae_y = 0

        self.w_all = []

    def add_realization(self, traj_xy, gt_xy):
        T = len(gt_xy)
        self.tot_ade_real += ade_xy(traj_xy, gt_xy); self.n_ade_real += 1
        self.tot_fde_real += fde_xy(traj_xy, gt_xy); self.n_fde_real += 1
        self.tot_mse_real += mse_xy(traj_xy, gt_xy); self.n_mse_real += 1

        # component-wise MAE (not ADE on 1D!)
        self.tot_mae_x += np.mean(np.abs(traj_xy[:, 0] - gt_xy[:, 0])); self.n_mae_x += 1
        self.tot_mae_y += np.mean(np.abs(traj_xy[:, 1] - gt_xy[:, 1])); self.n_mae_y += 1

    def add_meantraj(self, mean_xy, gt_xy):
        self.tot_ade_mean += ade_xy(mean_xy, gt_xy); self.n_ade_mean += 1
        self.tot_fde_mean += fde_xy(mean_xy, gt_xy); self.n_fde_mean += 1
        self.hit_list.append(hit(mean_xy, gt_xy))

    def add_multimodal(self, min_ade_k_val, min_fde_k_val):
        self.min_ade_list.append(min_ade_k_val)
        self.min_fde_list.append(min_fde_k_val)
    
    def add_most_likely(self, ml_ade_val, ml_fde_val):
        self.ml_ade_list.append(ml_ade_val)
        self.ml_fde_list.append(ml_fde_val)

    def add_w(self, w):
        self.w_all.append(w)

    def finalize(self):
        out = {}
        out['ade_real'] = self.tot_ade_real / max(self.n_ade_real, 1)
        out['fde_real'] = self.tot_fde_real / max(self.n_fde_real, 1)
        out['mse_real'] = self.tot_mse_real / max(self.n_mse_real, 1)
        out['rmse_real'] = np.sqrt(out['mse_real'])
        out['hit_rate'] = np.mean(self.hit_list) if len(self.hit_list) else np.nan
        out['ade_mean'] = self.tot_ade_mean / max(self.n_ade_mean, 1)
        out['fde_mean'] = self.tot_fde_mean / max(self.n_fde_mean, 1)
        out['ml_ade'] = np.mean(self.ml_ade_list) if len(self.ml_ade_list) else np.nan
        out['ml_fde'] = np.mean(self.ml_fde_list) if len(self.ml_fde_list) else np.nan
        out['min_ade_k'] = np.mean(self.min_ade_list) if len(self.min_ade_list) else np.nan
        out['min_fde_k'] = np.mean(self.min_fde_list) if len(self.min_fde_list) else np.nan
        out['mae_x'] = self.tot_mae_x / max(self.n_mae_x, 1)
        out['mae_y'] = self.tot_mae_y / max(self.n_mae_y, 1)
        return out

# -------------------
# Main
# -------------------
def main():
    dfd = pd.read_pickle(PICKLE)
    scenario_keys = sorted(set(dfd['scenario_file'].values))
    num_scenarios = len(scenario_keys)

    if MODE == 'iter':
        acc = MetricAccumulator()
        skipped = 0
        for i, key in enumerate(scenario_keys):
            data = dfd[dfd['scenario_file'] == key]
            x_set, y_set = [], []
            acc_set, dpsip_set = [], []
            acc_gt_set, dpsip_gt_set = [], []

            # generate samples
            nan_found = False
            for s in range(min(NUM_SAMPLES_PER_SCENARIO, len(data))):
                axp    = np.asarray(data['ax_pred'].values[s])
                dpsip  = np.asarray(data['dpsi_pred'].values[s])
                vx0    = data['v0'].values[s]
                psi0   = data['psi_0'].values[s]

                x_pred, y_pred = vmm(axp, dpsip, v_init=vx0, psi_init=psi0)[:2]

                if np.isnan(x_pred).any() or np.isnan(y_pred).any():
                    nan_found = True
                    break

                x_set.append(np.asarray(x_pred))
                y_set.append(np.asarray(y_pred))
                acc_set.append(axp)
                dpsip_set.append(dpsip)

            if nan_found or len(x_set) == 0:
                skipped += 1
                continue

            # ground truth (one per scenario assumed)
            xgt = np.asarray(data['x_gt'].values[0])
            ygt = np.asarray(data['y_gt'].values[0])
            gt_xy = np.stack([xgt, ygt], axis=1)

            acc_gt_set.append(np.asarray(data['ax_gt'].values[0]))
            dpsip_gt_set.append(np.asarray(data['dpsi_gt'].values[0]))

            # mean trajectory over samples
            traj_mean_x = np.mean(x_set, axis=0)
            traj_mean_y = np.mean(y_set, axis=0)
            mean_xy = np.stack([traj_mean_x, traj_mean_y], axis=1)

            # one random realization
            idx_rand = random.randrange(len(x_set))
            real_xy = np.stack([x_set[idx_rand], y_set[idx_rand]], axis=1)

            # accumulate metrics
            acc.add_realization(real_xy, gt_xy)
            acc.add_meantraj(mean_xy, gt_xy)

            # multimodal clustering on the set of sample trajectories
            trajectories_xy = np.stack([np.stack([x_set[j], y_set[j]], axis=1) for j in range(len(x_set))])  # (S,T,2)
            labels, n_clusters, K_range, bic_scores = None, 0, None, None
            try:
                trajectoriesp, labels, n_clusters, K_range, bic_scores = gmm_clustering(x_set, y_set, gt_xy, max_clusters=K_MAX)
                acc.add_multimodal(
                    minADE_k(gt_xy, trajectoriesp, labels, n_clusters, K=K_MAX),
                    minFDE_k(gt_xy, trajectoriesp, labels, n_clusters, K=K_MAX),
                )
                acc.add_most_likely(
                    ade_most_likely(gt_xy, trajectoriesp, labels, n_clusters, K=K_MAX),
                    fde_most_likely(gt_xy, trajectoriesp, labels, n_clusters, K=K_MAX),
                )

                
            except Exception:
                # If clustering fails, just skip multimodal stats for this scenario
                pass
            
            acc.add_w(float(data['w'].values[0]))

            # per-iteration reporting/plots
            if MODE == "iteri":
                ade_r = ade_xy(real_xy, gt_xy)
                fde_r = fde_xy(real_xy, gt_xy)
                print(f"[{i+1}/{num_scenarios}] {key} | ADE_real={ade_r:.3f} | FDE_real={fde_r:.3f}")

           
        # post-hoc reporting/plots
        results = acc.finalize()
        done = num_scenarios - skipped

        print(f"\nProcessed {done} scenarios ({CFG}, S = {DDIM_STEPS}).\n")
        print(f"ade_real: {results['ade_real']:.3f}")
        print(f"fde_real: {results['fde_real']:.3f}")
        #print(f"mse_real: {results['mse_real']:.3f}")
        #print(f"rmse_real: {results['rmse_real']:.3f}")
        #print(f"hit_rate: {results['hit_rate']:.3f}")
        print(f"ade_mean: {results['ade_mean']:.3f}")
        print(f"fde_mean: {results['fde_mean']:.3f}")
        print(f"ade_ml: {results['ml_ade']:.3f}")
        print(f"fde_ml: {results['ml_fde']:.3f}")
        print(f"min_ade_k (K={K_MAX}): {results['min_ade_k']:.3f}")
        print(f"min_fde_k (K={K_MAX}): {results['min_fde_k']:.3f}")

    else:
        # Collect posthoc bundles if needed
        print(set(list(dfd['q_idx'])))  # as you had
        cfgl = []
        var = []
        for q_idx in set(list(dfd['q_idx'])[:2]):
            df_group = dfd[dfd['q_idx'] == q_idx]
            idxs = df_group.index[:40]
            print(idxs)
            all_x, all_y, all_xgt, all_ygt = [], [], [], []
            y_last = []

            for i in idxs:
                axp  = np.array(df_group.loc[i, 'ax_pred'])
                dps  = np.array(df_group.loc[i, 'dpsi_pred'])
                v0   = df_group.loc[i, 'v0']
                psi0 = df_group.loc[i, 'psi_0']

                x_pred, y_pred = vmm(axp, dps, v_init=v0, psi_init=psi0)[:2]
                x_gt, y_gt     = df_group.loc[i, 'x_gt'], df_group.loc[i, 'y_gt']
                cfg            = df_group.loc[i, 'w']

                all_x.append(np.asarray(x_pred))
                all_y.append(np.asarray(y_pred))
                all_xgt.append(np.asarray(x_gt))
                all_ygt.append(np.asarray(y_gt))
                y_last.append(y_pred[-1])
                cfgl.append(round(float(cfg), 1))

            var.append(np.var(y_last))
            csv_dict = {}
            for k, i in enumerate(idxs):
                csv_dict[f"all_x_{i}"]  = all_x[k]
                csv_dict[f"all_y_{i}"]  = all_y[k]
                csv_dict[f"x_pred_{i}"] = all_x[k]   # same data you call x_pred
                csv_dict[f"y_pred_{i}"] = all_y[k]   # same data you call y_pred

            # Convert to DataFrame (each key → column)
            df_csv = pd.DataFrame(csv_dict)
            df_csv.to_csv(f"qidx_{q_idx}_trajectories.csv", index=False)
            
            # Save per q_idx so files do not overwrite
            df_csv.to_csv(f"qidx_{q_idx}_trajectories.csv", index=False)


if __name__ == "__main__":
    main()
