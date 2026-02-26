"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    V          = enum.auto()  

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        guidance = True,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.guidance = guidance
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    

    def q_posterior_mean_variance_sampling(self, x_t, t, eps):
        beta_t = _extract_into_tensor(self.betas, t, x_t.shape)
        alpha_t = 1.0 - beta_t
        alpha_bar_t = _extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)

        # Posterior variance (DDPM ancestral)
        posterior_variance = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * beta_t
        posterior_log_variance = th.log(posterior_variance.clamp(min=1e-20))

        # model-predicted mean μ_t using ε prediction
        model_mean = (
            1 / th.sqrt(alpha_t)
            * (x_t - (beta_t / th.sqrt(1 - alpha_bar_t)) * eps)
        )
        
        return model_mean, posterior_variance,posterior_log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON, ModelMeanType.V]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            elif self.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            else:  # V
                pred_xstart = process_xstart(self._predict_xstart_from_v(x_t=x, t=t, v=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape

        sqrt_alphas_cumprod_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)               # √(ᾱ_t)
        sqrt_one_minus_alphas_cumprod_t = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)  # √(1−ᾱ_t)

        # x0 = (x_t − √(1−ᾱ_t) · eps) / √(ᾱ_t)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * eps) / sqrt_alphas_cumprod_t

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_xstart_from_v(self, x_t, t, v):
        a  = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)          # √ᾱ_t
        b  = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)# √(1-ᾱ_t)
        return a * x_t - b * v

    def _predict_eps_from_v(self, x_t, t, v):
        a  = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
        b  = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return b * x_t + a * v


    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    
    def _ddim_sigma(self, t, shape, eta):
        alpha_bar     = _extract_into_tensor(self.alphas_cumprod, t, shape)
        alpha_bar_prev= _extract_into_tensor(self.alphas_cumprod_prev, t, shape)
        return (
            eta
            * th.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
            * th.sqrt(1.0 - alpha_bar / alpha_bar_prev)
        )
    

    def ddim_sample_cond(
        self, model, x, t, t_prev, guidance_scale, clip_denoised=True, denoised_fn=None,
        model_kwargs=None, eta=0.0, guidance_rescale=True,
    ):
        # 1) ε with classifier-free guidance
        B, C = x.shape[:2]
        assert t.shape == (B,)
        y_u = th.ones(B, device=x.device, dtype=th.long) * -1
        assert (y_u != model_kwargs.get("y", y_u)).any(), "cond and uncond labels identical!"
        # ----- forward (cond & uncond) -----
        out_u = model(x, self._scale_timesteps(t), **({**(model_kwargs or {}), 'y': y_u}))
        out_c = model(x, self._scale_timesteps(t), **(model_kwargs or {}))
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert out_u.shape == (B, C * 2, *x.shape[2:])
            out_u, _ = th.split(out_u, C, dim=1)
            out_c, _ = th.split(out_c, C, dim=1)

        # map model output → ε for CFG, regardless of native head
        if self.model_mean_type == ModelMeanType.V:
            eps_u = self._predict_eps_from_v(x_t=x, t=t, v=out_u)
            eps_c = self._predict_eps_from_v(x_t=x, t=t, v=out_c)
        elif self.model_mean_type == ModelMeanType.EPSILON:
            eps_u, eps_c = out_u, out_c
        else:
            raise NotImplementedError("DDIM CFG expects EPSILON or V parameterization")
        

        # CFG in ε-space
        g = eps_c - eps_u
        eps_g = eps_u + guidance_scale * g
       
        # x0 from guided ε
        def process_x0(z):
            if denoised_fn is not None: z = denoised_fn(z)
            return z.clamp(-1, 1) if clip_denoised else z
        pred_xstart = process_x0(self._predict_xstart_from_eps(x_t=x, t=t, eps=eps_g))

        
        # 3) DDIM update (Eq. 12)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod, t_prev, pred_xstart.shape)
        alpha_bar_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) \
                 * th.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        
        scale_term = (1 - alpha_bar_prev - sigma**2).clamp(min=0.0)
        mean_pred = (
            th.sqrt(alpha_bar_prev) * pred_xstart
            + th.sqrt(scale_term) * eps_g
        )

        nonzero_mask = (t_prev != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        noise = th.randn_like(x)
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": pred_xstart}

    def ddim_sample_loop(
        self,
        model,
        shape,
        guidance_scale,
        guidance_rescale,
        eta,
        ddim_steps,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            guidance_scale,
            guidance_rescale=guidance_rescale,
            eta=eta,
            ddim_steps = ddim_steps,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def _make_ddim_timesteps(self, num_sample_steps: int):
        # Create the time steps, from T-1 down to 0
        c_idx = np.linspace(0, self.num_timesteps - 1, num=num_sample_steps, dtype=np.int64)[1:]
        # Reverse the order for the descending sampling loop
        idx = np.asarray(c_idx, dtype=np.int64)[::-1] 
        # The previous time steps (t-1) for DDIM equation. t_0 is 0.
        prev_idx = np.append(np.asarray(c_idx, dtype=np.int64)[::-1][1:], 0) 
        return idx, prev_idx # returns (t, prev_t) pairs

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        guidance_scale,
        guidance_rescale,
        eta,
        ddim_steps,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False
    ):

        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        tau_list, tau_prev_list = self._make_ddim_timesteps(ddim_steps)  # <--- NEU
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            tau_list = tqdm(tau_list)

        def beta_schedule(T, gamma=2.0, start=0.1, end=1.0, eps=1e-6):
            t = th.linspace(0.0 + eps, 1.0 - eps, T)
            pdf = (t**(gamma - 1.0)) * ((1.0 - t)**(gamma - 1.0))
            pdf = pdf / pdf.max()
            return start + (end - start) * pdf

        def cosine_schedule(T, start=0.1, end=1.0):
            # slow rise early, quicker late
            x = (1 - th.cos(th.linspace(0, th.pi, T))) / 2
            return start + (end - start) * x
        
        #cfg_array = beta_schedule(len(tau_list))
        cfg_array = cosine_schedule(len(tau_list))

        for i, t_i in enumerate(tau_list):
            t = th.tensor([t_i] * shape[0], device=device)
            t_prev = th.tensor([tau_prev_list[i]] * shape[0], device=device)

            with th.no_grad():
                out = self.ddim_sample_cond(
                    model,
                    img,
                    t,
                    t_prev,
                    guidance_scale=cfg_array[i],
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    guidance_rescale=guidance_rescale,
                )

                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, lambda_smooth, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
       
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
            
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.V: (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
                - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start),
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            
            # ------------------------------------------------------------------
            # 🔧 Smoothness regularization (for temporal continuity)
            # ------------------------------------------------------------------
            if self.model_mean_type == ModelMeanType.EPSILON:
                x0_hat = self._predict_xstart_from_eps(x_t, t, model_output)
            elif self.model_mean_type == ModelMeanType.START_X:
                x0_hat = model_output
            elif self.model_mean_type == ModelMeanType.V:
                x0_hat = self._predict_xstart_from_v(x_t, t, model_output)
            else:
                x0_hat = self._predict_xstart_from_xprev(x_t, t, model_output)


            alphas_cumprod = _extract_into_tensor(self.alphas_cumprod, t, x0_hat.shape)
            eps = 1e-6
            # 1) normalize per sample/channel over time
            x0n = (x0_hat - x0_hat.mean(dim=-1, keepdim=True)) / (x0_hat.std(dim=-1, keepdim=True) + eps)
            # 2) robust TV on time diffs
            diff = x0n[..., 1:] - x0n[..., :-1]
            tv = (diff.pow(2) + 1e-6).sqrt()          # Charbonnier (robust L1)
            diff = x0_hat[..., 1:] - x0_hat[..., :-1]
            terms["smooth"] = mean_flat(tv)

            
            geometric_loss=False
            if geometric_loss :
                terms['geom_loss'] =0
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"] + lambda_smooth * terms["smooth"] 
            else:
                terms["loss"] = terms["mse"] + lambda_smooth * terms["smooth"] 
        else:
            raise NotImplementedError(self.loss_type)
        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

'''

    def p_sample_cond(
        self, model, x, t, guidance_scale, clip_denoised=True, denoised_fn=None, model_kwargs=None,
        guidance_rescale=False
    ):
        """
        DDPM-Step mit Classifier-Free Guidance im ε-Raum.
        guidance_scale: float oder None (falls None -> 0).
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)

        # 1) ε mit CFG
         # ----- forward (cond & uncond) -----
        y_u = th.ones(B, device=x.device, dtype=th.long) * -1
        eps_u = model(x, self._scale_timesteps(t), **({**model_kwargs, 'y': y_u}))
        eps_c = model(x, self._scale_timesteps(t), **(model_kwargs or {}))

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert eps_u.shape == (B, C * 2, *x.shape[2:])
            eps_u, _ = th.split(eps_u, C, dim=1)
            eps_c, _ = th.split(eps_c, C, dim=1)

        # ----- build guided x0 prediction -----
        def process_x0(z):
            if denoised_fn is not None: z = denoised_fn(z)
            return z.clamp(-1, 1) if clip_denoised else z
        
        if self.model_mean_type == ModelMeanType.EPSILON:
            # model outputs epsilon
            g = eps_c - eps_u
            if guidance_rescale:
                dims = list(range(1, g.ndim))
                scale = guidance_scale * (eps_u.norm(dim=dims, keepdim=True) / (g.norm(dim=dims, keepdim=True) + 1e-8))
                eps_g = eps_u + scale * g
            else:
                eps_g = eps_u + guidance_scale * g
            pred_xstart = process_x0(self._predict_xstart_from_eps(x_t=x, t=t, eps=eps_g))
        else:
            raise NotImplementedError(self.model_mean_type)

        # 3) Posterior-Parameter (not true but estimated one)
        print('prev has to be implemented')
        model_mean, posterior_variance, _ = self.q_posterior_mean_variance_sampling(x_t=x, t=t, eps=eps_g)
        
        # 4) DDPM-Sampling-Rauschen
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = model_mean + nonzero_mask * th.sqrt(posterior_variance) * noise
        return {"sample": sample, "pred_xstart": pred_xstart}



    def p_sample_loop(
        self,
        model,
        shape,
        guidance_scale,
        guidance_rescale,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            guidance_scale=guidance_scale,
            guidance_rescale = guidance_rescale,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        guidance_scale,
        guidance_rescale,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                if guidance_scale is not None:
                    out = self.p_sample_cond(
                        model,
                        img,
                        t,
                        guidance_scale,
                        guidance_rescale = guidance_rescale,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                    )
                else:
                    out = self.p_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                    )
                yield out
                img = out["sample"]

'''