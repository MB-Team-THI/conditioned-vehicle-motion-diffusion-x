"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import pickle
import yaml
import argparse
import os
import pandas as pd
from improved_diffusion.vmm import *
import scipy
import sys
import numpy as np
import torch as th
import torch.distributed as dist
from improved_diffusion.image_datasets import load_data
from mahalanobis.cfg_weighter import *

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.norm_utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

config_path = 'configs/config.yaml'
ITER = 1 #Number of iterations
TEST_DIR = '../datasets/highD/test'

class NP2toNP1Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect numpy._core.* -> numpy.core.*
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)
    
def set_path(model_path, folder_name='resultsNEWk'):
    dir_name = os.path.dirname(model_path)
    out_dir = os.path.join(dir_name, folder_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def progress_bar(current, total, bar_length=30):
    fraction = current / total
    arrow = '=' * int(fraction * bar_length - 1) + '🚀'
    padding = ' ' * (bar_length - len(arrow))
    end_char = '\r' if current < total else '\n'
    print(f'Progress: 🌍[{arrow}{padding}]🌘 {current}/{total}', end=end_char)
    sys.stdout.flush()

def print_gen_info(args):
    """
    Print formatted information about the generative process.
    """
    # Extract file name (ema)
    ema_name = os.path.basename(args.model_path)

    # Extract parent folder name (checkpoint folder)
    ckpt_folder = os.path.basename(os.path.dirname(args.model_path))

    print("\n" + "=" * 70)
    print("⚗️  Starting Generative Process")
    print("=" * 70)
    # Print key arguments — adapt this list to what’s relevant for you
    print(f"Model name:           {ema_name}")
    print(f"Model folder:         {ckpt_folder}")
    print(f"Condition file:       {os.path.basename(args.cond_path)}")
    print(f"Codbook entries:      {args.num_classes}(-1)")
    print("=" * 70 + "\n")

def main():
    args = create_argparser().parse_args()
    out_dir = set_path(args.model_path)
    dist_util.setup_dist()
    device = dist_util.dev()
    logger.configure(out_dir)

    pklfile = args.cond_path
    with open(pklfile, "rb") as f:
        vqdf = NP2toNP1Unpickler(f).load()

    print_gen_info(args)

    #logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
        
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    #logger.log("sampling...")

    
    test_dir = TEST_DIR
    data = load_data(data_dir=test_dir, batch_size=args.batch_size, class_cond=args.class_cond,)

    df_cols = ['scenario_file', 'sample_id', 'q_idx', 'x_gt', 'y_gt', 'ax_gt', 
               'dpsi_gt', 'ax_pred', 'dpsi_pred', 'v0', 'psi_0', 'd', 'w']
    df = pd.DataFrame(columns=df_cols)

    weighter = MahalanobisWeighter(wmin=args.guidance_scale_min, 
                                   wmax=args.guidance_scale_max, 
                                   mth=args.mahanalobi_dist_max)
    
    iter = 0
    args_ddim_steps = args.ddim_steps
    id_to_qidx = dict(zip(vqdf['scenario_id'], vqdf['q_idx']))  # 1) fast lookup
    for batch, _ in data:
        if iter >= ITER/(args.batch_size):  # or your dataset length // batch_size
            break
        progress_bar(iter*args.batch_size, ITER)
        model_kwargs = {}
        files = batch['file']
        scenario_ids = batch['scenario_id']           # list[str] of length B
        q_idx_list = [id_to_qidx.get(sid, -1) for sid in scenario_ids]
        q_idx = th.as_tensor(q_idx_list, dtype=th.long, device=device)
        # to kwarg
        model_kwargs = {"y": q_idx}

        
        # group truth
        x_gt = batch['predicted_x'][:,0,:].to(device)
        y_gt = batch['predicted_y'][:,0,:].to(device)
        ax_gt = batch['predicted_ax'][:,0,:].to(device)
        dspi_gt = batch['predicted_dpsi'][:,0,:].to(device)
        psi0 = batch['psi_0'][:,0,0].reshape(-1,1).to(device)
        vx0 = batch['vx0'].reshape(-1,1).to(device)
        

        # optional classifier-free guidance weight via Mahalanobis distance
        d = None
        w= args.guidance_scale_max
        wg = th.ones((args.batch_size, 2, 125), device=device) * args.guidance_scale_max
        # Here scale with mahanalobis distance
        '''if diffusion.guidance:
            u = vqdf[vqdf['scenario_id'].isin(batch['scenario_id'])]['q_vec']
            u = np.vstack(u.values)[0]
            v = vqdf[vqdf['scenario_id'].isin(batch['scenario_id'])]['z_vec']
            v = np.vstack(v.values)[0]
            S = vqdf[vqdf['scenario_id'].isin(batch['scenario_id'])]['q_cov']
            S = np.stack(S.values)[0]
            d = scipy.spatial.distance.mahalanobis(u,v,S)
            w = weighter.compute(d)'''
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)  
        
        for sid in range(0, args.num_samples):
            if args.use_ddim:
                sample = sample_fn(
                    model,
                    (args.batch_size, 2, args.image_size),
                    guidance_scale = wg,
                    guidance_rescale= args.guidance_rescale,
                    eta = args.eta,
                    ddim_steps = args.ddim_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            else:
                sample = sample_fn(
                    model,
                    (args.batch_size, 2, args.image_size),
                    guidance_scale = w,
                    guidance_rescale= args.guidance_rescale,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
           
            if model.use_vmm:
                sampleN =  inv_vmm_norm(sample)
            else:
                sampleN = inv_xy_norm(sample)

            with th.no_grad():
                # bring referenced tensors to CPU for pandas storage
                x_gt_np = x_gt.detach().cpu().numpy()
                y_gt_np = y_gt.detach().cpu().numpy()
                psi0_np = psi0.detach().cpu().numpy()
                vx0_np  = vx0.detach().cpu().numpy()
                samp_np = sampleN.detach().cpu().numpy()   # [B,2,T]
                ax_gt_np = ax_gt.detach().cpu().numpy()
                dspi_gt_np = dspi_gt.detach().cpu().numpy()

            for i in range(samp_np.shape[0]):
                ndf = pd.DataFrame({
                    'scenario_file': [files[i]],
                    'sample_id': [sid],
                    'q_idx': [int(q_idx.detach().cpu().numpy()[i])],
                    'x_gt': [x_gt_np[i, :]],
                    'y_gt': [y_gt_np[i, :]],
                    'ax_gt': [ax_gt_np[i, :]],
                    'dpsi_gt': [dspi_gt_np[i, :]],
                    'ax_pred': [samp_np[i, 0, :]],
                    'dpsi_pred': [samp_np[i, 1, :]],
                    'v0': [float(vx0_np[i, 0])],
                    'psi_0': [float(psi0_np[i, 0])],
                    'd': [None if d is None else float(d)],
                    'w': [None if w is None else float(w)],
                })
                df = pd.concat([df, ndf], ignore_index=True)
        iter += 1
    # ----- save results (per rank to avoid write collisions) -----
    # encode cfg value safely for filename
    if args.guidance_rescale:
        out_base = os.path.join(logger.get_dir(), f"results_{args_ddim_steps}_gencfg={w}r")
    else:
        out_base = os.path.join(logger.get_dir(), f"results_{args_ddim_steps}_gencfg={w}")
    df.to_pickle(out_base + ".pkl")

def create_argparser():
    def load_config(yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    cfg = load_config(os.path.join(os.getcwd(), config_path))["generate"]

    # Model
    fpath = cfg["folder_path"]
    mfolder = cfg["model_folder"]
    modelname = cfg["model_name"]
    model_path = os.path.join(fpath, mfolder, modelname)
    # Conditions
    base_test_file = cfg["cond_file_test"]  # '10_22_ce60_cl1_train_results'
    cond_dir = cfg["cond_dir"]
    cond_file_test_pkl = os.path.join(cond_dir, base_test_file + ".pkl")

    defaults = dict(
        model_path= model_path,
        cond_path = cond_file_test_pkl,
        clip_denoised=cfg["clip_denoised"],
        num_samples=cfg["num_samples"],
        batch_size=cfg["batch_size"],
        use_ddim=cfg["use_ddim"],
        eta=cfg["eta"],
        ddim_steps=cfg["ddim_steps"],
        guidance_rescale=cfg["guidance_rescale"],
        guidance_scale_min=cfg["guidance_scale_min"],
        guidance_scale_max=cfg["guidance_scale_max"],
        mahanalobi_dist_max=cfg["mahanalobi_dist_max"],
    )
    num_classes = cfg["num_classes"] +1 # incl default zero
    predict_v = cfg["predict_v"]
    defaults.update(model_and_diffusion_defaults(num_classes = num_classes, predict_v=predict_v))

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

