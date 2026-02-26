import yaml
import argparse
import os
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.nn import count_parameters
import time
import datetime
import tempfile
import json

config_path = 'configs/config.yaml'

def main():
    args = create_argparser().parse_args()
    run_path = os.path.join(os.getcwd(), 'ckpts',datetime.datetime.now().strftime("ckpt-%Y-%m-%d-%H-%M-%S-%f"))
    
    dist_util.setup_dist()
    logger.configure(run_path)
    with open(os.path.join(run_path,'run_args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    logger.log('Number of model params: ' +str(count_parameters(model)))
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(data_dir=args.data_dir, batch_size=args.batch_size, class_cond=args.class_cond,)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        lambda_smooth = args.lambda_smooth,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        cond_dropout_rate = args.cond_dropout_rate,
        cond_path = args.cond_path,
        use_lr_scheduler= args.use_lr_scheduler,
        warmup_steps= args.warmup_steps,
        min_lr= args.min_lr,
        warmup_start_factor=args.warmup_start_factor,
    ).run_loop()


def create_argparser():
    def load_config(yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    cfg = load_config(os.path.join(os.getcwd(), config_path))["train"]
    data_dir = os.path.join(os.getcwd(), "datasets" +os.sep + cfg["data_name"] + os.sep +"train")
    cond_path = os.path.join(cfg["cond_dir"], cfg["cond_file"] + ".csv")

    defaults = dict(
        data_dir= data_dir,
        schedule_sampler=cfg["schedule_sampler"],
        lr=cfg["lr"],
        lambda_smooth=cfg["lambda_smooth"],
        cond_dropout_rate=cfg["cond_dropout_rate"],
        weight_decay=cfg["weight_decay"],
        lr_anneal_steps=cfg["lr_anneal_steps"],
        batch_size=cfg["batch_size"],
        microbatch=cfg["microbatch"],
        ema_rate=cfg["ema_rate"],
        log_interval=cfg["log_interval"],
        save_interval=cfg["save_interval"],
        resume_checkpoint=cfg["resume_checkpoint"],
        use_fp16=cfg["use_fp16"],
        fp16_scale_growth=cfg["fp16_scale_growth"],
        cond_path= cond_path,
        use_lr_scheduler=cfg["use_lr_scheduler"],
        warmup_steps=cfg["warmup_steps"],
        min_lr=cfg["min_lr"],
        warmup_start_factor=cfg["warmup_start_factor"],
    )
    num_classes = cfg["num_classes"] +1 # incl default zero
    predict_v = cfg["predict_v"]
    defaults.update(model_and_diffusion_defaults(num_classes = num_classes, predict_v=predict_v))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
