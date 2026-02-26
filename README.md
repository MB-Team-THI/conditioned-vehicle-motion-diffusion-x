# cVMDx: Improved Conditioned Vehicle Motion Diffusion Model

[![arXiv](https://img.shields.io/badge/arXiv-<2602.21319>-<COLOR>.svg)](https://arxiv.org/abs/2602.21319)

The official PyTorch implementation of the paper <br />
[**"Uncertainty-Aware Diffusion Model for Multimodal Highway Trajectory Prediction via DDIM Sampling"**](https://arxiv.org/abs/2602.21319).
<img width="2803" height="847" alt="image" src="https://github.com/user-attachments/assets/32e4f7f1-9a8c-484e-921b-8ea551a0d16e" />
[arXiv](https://arxiv.org/abs/2602.21319) | [BibTeX](#bibtex) 

This repository contains only the code for the `Vehicle Motion Diffusion` module. 
Note, that the adaptive cfg scale computation via UQ is disabled in the default setting. 

The implementation of the `Context Conditioning` module via CVQ-VAE can be found in the completeness-pipeline repository:
https://github.com/mb-team-thi/completeness-pipeline

## Getting started

This code was tested on `Ubuntu 22.04.4 LTS` and requires:

* Python 3.8.5
* conda 
* CUDA capable GPU (one is enough)<br />

### 1. Setup environment

Install the OpenMPI development libraries:
```shell
sudo apt-get install libopenmpi-dev
```

Setup conda env:
```shell
conda env create -f environment.yaml
conda activate cvmdx
```
<br />


Required/recommended repository structure:
```
cvmdx
├── ckpts (folder_path)
│   ├── model_folder
|   │   ├── model_name.pt
|   │   ├── ...
├── configs
│   ├── config.yaml
├── datasets
│   ├── highD
|   │   ├── train
|   │   ├── test
├── cond_dir/
│   ├── cond_file.csv          
│   ├── cond_file_test.csv    
├── improved_diffusion
|   ├── ...
├── mahalanobis_distance
|   ├── ...
├── scripts
|   ├── ...
├── ...
├── evaluate_trajectories.py
├── generate_trajectories.py
├── train_trajectories.py
├── environment.yaml
```


### 2. Get data
The official implementation was tested on the highD dataset. However, also other publicly available datasets can be used.

NOTE: Please be aware that the data examples in this repository are not taken from the highD or any other dataset. Instead, they are computer-generated for illustrative purposes. The features may not necessarily reflect actual, natural vehicle behavior.

<details>
  <summary><b>highD</b></summary>

To get the data follow the steps provided on the dataset homepage: https://levelxdata.com/highd-dataset/.

</details>
<details>
  <summary><b>NGSIM</b></summary>

To get the data follow the steps provided on the dataset homepage: https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm.

</details>
<details>
  <summary><b>Automatum data</b></summary>

To get the data follow the steps provided on the dataset homepage: https://automatum-data.com/de.

</details>
<br />


### 3. Prepare Data
Within the `datasets` folder generate a new folder for each dataset, e.g. highD.
```
├── datasets
│   ├── example_dataset
│   ├── highD
│   ├── ...
```
Each `dataset` subfolder needs to follow this hierarchical structure
```
datasets/highD/
├── train
│   ├── class0
|   │   ├── kl0.mat
|   │   ├── ...
│   ├── class1
|   │   ├── lcr0.mat
|   │   ├── ...
│   ├── class2
|   │   ├── lcl0.mat
|   │   ├── ...
├── test
│   ├── class0
|   │   ├── ...
│   ├── class1
|   │   ├── ...
│   ├── class2
|   │   ├── ...
├── ...
```
* `class0` Holds scenarios classifies as keep lane (kl) scenarios.
* `class1` Holds scenarios classifies as lange change right (lcr) scenarios.
* `class2` Holds scenarios classifies as lange change left  (lcl) scenarios.


xxx.mat file structure

```
xxx.mat
  |    - keys -                          - type -                 - size -
  ├── 'data_keys'                        Array of strings          1 x N
  └── 'observed_data_x'                  Array of float64          N x T_o
  └── 'observed_data_y'                  Array of float64          N x T_o
  └── 'observed_data_vx'                 Array of float64          N x T_o
  └── 'observed_data_vy'                 Array of float64          N x T_o
  └── 'scenario_type'                    Array of strings          1 x 1
  └── 'predicted_x'                      Array of float64          1 x T_p
  └── 'predicted_y'                      Array of float64          1 x T_p
  └── 'predicted_ax'                     Array of float64          1 x T_p
  └── 'predicted_dpsi'                   Array of float64          1 x T_p
  └── 'psi_0'                            Array of float64          1 x 1
  └── 'v0'                               Array of float64          1 x 1
```
Dimensions:
* `N` - Maximal number of considered vehicles (incl. ego).
* `T_o` - Number of observation steps (t_obs x f, e.g : 3s x 25 Hz = 75 steps).
* `T_p` - Number of prediction steps (t_pred x f, e.g : 5s x 25 Hz = 125 steps).
  
```
'data_keys' = ['ego ', 'following ', 'preceding',
       'leftPreceding ', 'leftAlongside ', 'leftFollowing ',
       'rightPreceding', 'rightAlongside', 'rightFollowing'] 
```
```
'scenario_type' = ['keep_lane']  or  ['lane_change_left']  or ['lane_change_right']
```

<br />

### 4. Condition Files (Prerequisite)
A prerequisite for training and generation is that the conditioning files are correctly prepared and stored in the condition directory.
The directory specified by `--cond_dir` must exist and contain the condition files for training and testing:
```
cond_dir/
├── cond_file.csv          
├── cond_file_test.csv     
```
* `cond_file.csv` corresponds to the value of cond_file in the config and is used during training.
* `cond_file_test.csv` corresponds to the value of cond_file_test in the config and is used during testing or generation.
Both files must be readable CSV files and follow the expected internal structure.

| Column name   | Type                       | Description                               | Notes / Constraints                                  |
| ------------- | -------------------------- | ----------------------------------------- | ---------------------------------------------------- |
| `scenario_id` | string                     | Unique identifier for a scenario                | Arbitrary string ID; not necessarily numeric         |
| `q_idx`       | integer                    | Assigned codebook entry/identifier of scneario  | Typically ranges from 0 to N−1                       |
| `q_vec`       | string (serialized vector) | Codebook entry embedding vector                 | Stored as a string representation of a numeric array |
| `z_vec`       | string (serialized vector) | Latent embedding vector of scenario             | Stored as a string representation of a numeric array |
| `diff`        | float                      | Distance metric d(q_vec, z_vec), e.g., Euclidean distance        | Must be numeric                                      |
| `q_mu`        | string (serialized vector) | Mean vector of all `z_vec` associated with `q_idx`             | One of a fixed set of categorical vectors            |
| `q_cov`       | string (serialized matrix) | Covariance matrix  of all `z_vec` associated with `q_idx`       | One of a fixed set of categorical matrices           |


### 5. Configuration
  
#### 1 ) Define your training configurations
Adapt the desired training parameters in the file 'configs/config.yaml':
* `data` defines dataset + conditioning sources (shared defaults).
* `train` defines training parameters like optimization, scheduling, EMA/logging/checkpointing and conditioning dropout.
* `generate` defines testing/generation parameters, e.g., which checkpoint to load and how sampling/guidance is performed.


**Data**
* Use `--data_name` to define the dataset name/identifier (here: highD).
* Use `--cond_dir`  to define the directory where the conditioning data is stored (e.g., VQ-VAE codebook assignments). This is the base folder used to locate both train/test condition files.
* Use `--cond_file` to define the filename (or run identifier) of the conditioning results used for training (.csv-file). 
* Use `--cond_file_test` to define the filename (or run identifier) of the conditioning results used for testing/evaluation (.csv-file).
* Use `--num_classes` to define the number of discrete conditioning classes (typically the VQ codebook size); e.g., with num_classes=256, the conditioning indices are expected to be in [0, 255].


**Train**
* Use `--schedule_sampler` to define how diffusion timesteps are sampled during training. uniform means each timestep is equally likely.
* Use `--lr` to set the learning rate for the optimizer (here: 2e-4).
* Use `--lambda_smooth` to weight an additional “smoothness” regularization term (0 disables this loss contribution).
* Use `--cond_dropout_rate` to set the probability of dropping the conditioning signal during training (classifier-free style conditioning dropout). 0.2 means 20% of training examples become “unconditioned”.
* Use `--weight_decay` to apply L2 weight decay regularization in the optimizer (0 disables).
* Use `--lr_anneal_steps` to define the total number of steps over which learning rate annealing is applied (here: 80,000 steps).
* Use `--use_lr_scheduler` to enable/disable the learning-rate scheduler logic (true enables it).
* Use `--warmup_steps` to define the number of optimizer steps used to linearly warm up the learning rate at the start (here: 3,000 steps).
* Use `--min_lr to define` the minimum learning rate floor used by the scheduler (commonly for cosine schedules).
* Use `-warmup_start_factor` to define the initial warmup LR as a fraction of lr (here: 1% of lr at step 0, then ramps up).
* Use `-batch_size` to define the global batch size used for training (here: 256).
* Use `-predict_v` to switch the diffusion model parameterization to predict v (velocity) instead of ε/x0 (commonly improves stability/quality in some setups).
* Use `-microbatch` to enable gradient accumulation via splitting the batch into microbatches. -1 disables microbatching (process full batch at once).
* Use `-ema_rate` to define the exponential moving average decay(s) applied to model weights. A string like "0.9999" supports comma-separated values (multiple EMA tracks).
* Use `-log_interval` to define how often (in steps) training stats are logged (here: every 50 steps).
* Use `-save_interval` to define how often (in steps) checkpoints are saved (here: every 20,000 steps).
* Use `-resume_checkpoint` to define a checkpoint path to resume training from. Empty string means “start fresh”.
* Use `-use_fp16` to enable mixed precision (FP16) training (false disables).
* Use `-fp16_scale_growth` to control loss-scale growth behavior for FP16 training (only relevant if use_fp16: true).


**Generate**
* Use `--model_name` to define which checkpoint file to load for sampling (e.g., : ema_0.9999_080000.pt).
* Use `--model_folder` to define a subfolder/group name under your checkpoints directory.
* Use `--folder_path` to define the base path where checkpoints are stored; the final model path is typically something like folder_path/model_folder/model_name.
* Use `--clip_denoised` to enable/disable clipping of predicted denoised outputs to a valid range during sampling (true often improves stability, depending on data scaling).
* Use `--predict_v` to ensure sampling uses the same parameterization as training (true means the checkpoint expects v-prediction).
* Use `--num_samples` to define how many samples to generate.
* Use `--batch_size` to define the sampling batch size (often can be larger than training if memory allows; here: 256).
* Use `--use_ddim` to switch sampling from ancestral DDPM sampling to DDIM sampling (true enables faster/deterministic sampling depending on eta).
* Use `--eta` to control DDIM stochasticity. 0.0 is deterministic DDIM; higher values add noise (more stochastic).
* Use `--ddim_steps` to define the number of DDIM sampling steps (fewer steps = faster but potentially lower quality).
* Use `--guidance_rescale` to enable/disable guidance rescaling (commonly used to mitigate overexposure/overguidance artifacts). False disables it.
* Use `--guidance_scale_min` to define the minimum classifier-free guidance scale used during sampling (here: 0.1).
* Use `--guidance_scale_max` to define the maximum classifier-free guidance scale used during sampling (here: 1.0).
* Use `--mahanalobi_dist_max` to set a maximum Mahalanobis distance threshold ( used as a filter/constraint on generated samples or conditions; 50 sets how strict the cutoff is, depending on implementation).


### 6. Train model (DDPM only)

This repository only provides the vehicle motion diffusion model (conditioned trajectory generation/prediction).
The adaptive guidance scale computation for DDPM is not performed in the default mode. It is set to a fixed value.

#### 1 ) Define your training configurations
#### 2 ) Run training loop
Change directory to folder `cvmdx`:
```shell
cd cvmdx
```
Then run
```shell
python train_trajectories.py 
```

<br />

### 5. Test the trained model
#### 1 ) Define your generation configurations
#### 2 ) Run generation loop
If not done before, change directory to folder `cvmdx`:
```shell
cd cvmdx
```
Adapt the run parameters in the file `generate_trajectories.py` (captial letter variables).

Then run
```shell
python generate_trajectories.py 
```
#### 3 ) Run evaluation loop

If not done before, change directory to folder `cvmdx`:
```shell
cd cvmdx
```
Adapt the run parameters in the file `evaluate_trajectories.py` (captial letter variables).

Then run
```shell
python evaluate_trajectories.py 
```

<br />

## Acknowledgments

This code builds upon the impactful work of predecessors. We want to thank the following contributors
that our code is based on:
[cVMD](https://github.com/MB-Team-THI/conditioned-vehicle-motion-diffusion), [completeness-pipeline](https://github.com/mb-team-thi/completeness-pipeline), [improved-diffusion](https://github.com/openai/improved-diffusion), [VQ-GAN](https://github.com/CompVis/taming-transformers)

<br />

## License
This code is distributed under an [MIT LICENSE](LICENSE).
Note that the code depends on other libraries and use a dataset that each have their own respective licenses that must also be followed.

<br />

## BibTeX
If you find this code useful in your research, please cite:
```bibtex
@inproceedings{neumeier2026cvmdx,
  title     = {Uncertainty-Aware Diffusion Model for Multimodal Highway Trajectory Prediction via DDIM Sampling},
  author    = {Neumeier, Marion and Ro{\ss}berg, Niklas and Botsch, Michael and Utschick, Wolfgang},
  booktitle = {Proceedings of the IEEE Intelligent Vehicles Symposium (IV)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.21319}
}
