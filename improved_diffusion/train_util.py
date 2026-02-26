import copy
import functools
import os
import sys
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import pandas as pd
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim import lr_scheduler 


from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .norm_utils import vmm_norm, xy_norm
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        lambda_smooth,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        cond_dropout_rate = 0.0,
        cond_path = None,
         # --- NEW ---
        use_lr_scheduler=True,      # turn scheduler on/off
        warmup_steps=0,             # number of warmup *steps*
        min_lr=1e-6,                # cosine floor
        warmup_start_factor=1e-2,   # warmup start factor (relative to lr)
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.lambda_smooth = lambda_smooth
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.cond_path = cond_path
        self.cond_dropout_rate =cond_dropout_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        ####
        self.use_lr_scheduler = use_lr_scheduler
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.warmup_start_factor = float(warmup_start_factor)
        # Build a step-based warmup + cosine scheduler (SequentialLR)
        self.scheduler = None
        if self.use_lr_scheduler and self.lr_anneal_steps:
            total_steps = int(self.lr_anneal_steps)
            warmup = min(self.warmup_steps, total_steps)  # clamp safely
            cosine = max(total_steps - warmup, 1)

            warmup_scheduler = lr_scheduler.LinearLR(
                self.opt, start_factor=self.warmup_start_factor, total_iters=warmup
            )
            cosine_scheduler = lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=cosine, eta_min=self.min_lr
            )
            # milestones is the length of the *first* scheduler in steps
            self.scheduler = lr_scheduler.SequentialLR(
                self.opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup]
            )

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]
        
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06d}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
            # NEW: load scheduler if we have one
        if self.scheduler is not None:
            sched_checkpoint = bf.join(bf.dirname(main_checkpoint), f"sched{self.resume_step:06d}.pt")
            if bf.exists(sched_checkpoint):
                logger.log(f"loading scheduler state from checkpoint: {sched_checkpoint}")
                sched_state = dist_util.load_state_dict(
                    sched_checkpoint, map_location=dist_util.dev()
                )
                try:
                    self.scheduler.load_state_dict(sched_state)
                except Exception as e:
                    logger.log(f"warning: failed to load scheduler state: {e}")

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        vqdf = pd.read_csv(self.cond_path)
        id_to_qidx = dict(zip(vqdf['scenario_id'], vqdf['q_idx']))  # 1) fast lookup
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, _ = next(self.data)
            # vqvae cond
            #q_idx = vqdf[vqdf['scenario_id'].isin(batch['scenario_id'])]['q_idx'].to_numpy()
            scenario_ids = batch['scenario_id']           # list[str] of length B
            q_idx_list = [id_to_qidx.get(sid, -1) for sid in scenario_ids]
            cond = th.as_tensor(q_idx_list, dtype=th.long)
            
            if self.model.use_vmm:
                acc = batch['predicted_ax']
                dpsi = batch['predicted_dpsi']
                batch, _ = vmm_norm(acc, dpsi)
            else:
                x = batch['predicted_x']
                y = batch['predicted_y']
                batch,_ = xy_norm(x, y)

            # Random mask: True for the elements to drop
            drop_mask = th.rand(self.batch_size, device=cond.device) < self.cond_dropout_rate
            # Clone to avoid modifying in-place if cond is reused
            cond_dropped = cond.clone()
            cond_dropped[drop_mask] = -1
            cond = cond_dropped

            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                'sve interval'
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev()).type(th.float32)
            
            if cond is not None:
                cond = cond[i : i + self.microbatch].to(dist_util.dev())
                micro_cond={'y': cond}
            else :
                micro_cond = None 

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
         
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                self.lambda_smooth,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
         # REMOVE manual anneal if scheduler is active
        if self.scheduler is None:
            self._anneal_lr()

        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        if self.scheduler is None:
            self._anneal_lr()

        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            if (p is not None):
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)
        logger.logkv("lr", self.opt.param_groups[0]["lr"])

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
          # NEW: save scheduler (if any)
        if self.scheduler is not None:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"sched{(self.step+self.resume_step):06d}.pt"), "wb"
            ) as f:
                th.save(self.scheduler.state_dict(), f)
        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
