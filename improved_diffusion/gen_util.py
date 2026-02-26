import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th


class NP2toNP1Unpickler(pickle.Unpickler):
    """Compatibility layer redirecting numpy._core.* to numpy.core.* when unpickling."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def seed_everything(seed: int):
    if seed is None:
        return
    import random
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def progress_bar(current: int, total: int, bar_length: int = 30):
    current = min(current, total)
    frac = current / max(1, total)
    done = max(0, int(frac * bar_length) - 1)
    arrow = "=" * done + "🚀"
    pad = " " * (bar_length - len(arrow))
    end = "\r" if current < total else "\n"
    print(f"Progress: 🌍[{arrow}{pad}]🌘 {current}/{total}", end=end)
    sys.stdout.flush()


def print_gen_info(args):
    ema_name = os.path.basename(args.model_path)
    ckpt_folder = os.path.basename(os.path.dirname(args.model_path))
    print("\n" + "=" * 70)
    print("⚗️  Starting Generative Process")
    print("=" * 70)
    print(f"Model name:           {ema_name}")
    print(f"Model folder:         {ckpt_folder}")
    print(f"Condition file:       {os.path.basename(args.cond_path)}")
    print(f"Codebook entries:     {args.num_classes} (incl. null)")
    print(f"Batch size:           {args.batch_size}")
    print(f"Samples per item:     {args.num_samples}")
    if args.max_batches is not None:
        print(f"Max batches:          {args.max_batches}")
    print(f"DDIM:                 {args.use_ddim} "
          f"(steps={args.ddim_steps}, eta={args.eta})")
    print(f"Guidance rescale:     {args.guidance_rescale}")
    print("=" * 70 + "\n")

def set_path(model_path, folder_name='resultsNEWk'):
    dir_name = os.path.dirname(model_path)
    out_dir = os.path.join(dir_name, folder_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def make_output_dir(model_path: str, folder_name: str) -> Path:
    out_dir = Path(model_path).resolve().parent / folder_name
    return ensure_dir(out_dir)


def safe_to_cpu_np(t: th.Tensor):
    return t.detach().to("cpu", non_blocking=True).numpy()
