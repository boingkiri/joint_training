import jax
from jax.lib import xla_bridge
from jax import random

import wandb
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig

from framework.unifying_framework import UnifyingFramework

import os
import sys
import argparse

from jax_smi import initialise_tracking

# @hydra.main(config_path="configs", config_name="config")
def start(config: DictConfig):
    rng = random.PRNGKey(config.rand_seed)
    model_type = config.type

    # if the current environment is GPU, set the available GPU
    if xla_bridge.get_backend().platform == "gpu":
        if hasattr(config, "available_gpus"):
            os.environ["CUDA_VISIBLE_DEVICES"] = config.available_gpus
    
    initialise_tracking()

    if config.get("distributed_training", False):
        # Assume that the running environment is TPU
        # If there is a need to use GPU, the code should be modified: TODO
        jax.distributed.initialize()
        if jax.process_index() != 0:
            sys.stdout = open(os.devnull, 'w')
        rand_seed = config.rand_seed + jax.process_index()
        rng = random.PRNGKey(rand_seed)

    print("-------------------Config Setting---------------------")
    print(OmegaConf.to_yaml(config))
    print("------------------------------------------------------")
    diffusion_framework = UnifyingFramework(model_type, config, rng)


    if config.do_training:
        name = config['exp_name']
        tags = config["tags"]
        project_name = f"my-{config.type}-WIP"
        args ={
            "project": project_name,
            "name": name,
            "tags": tags,
            "config": {**config}
        }
        wandb.init(**args)
        print("Training selected")
        diffusion_framework.train()

    if config.do_sampling:
        print("Sampling selected")
        diffusion_framework.sampling_and_save(config.num_sampling)
        fid_score = diffusion_framework.fid_utils.calculate_fid(config.exp.sampling_dir)
        print(fid_score)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffuion")
    parser.add_argument("--config", action="store", type=str, default="config")
    parser.add_argument("--only_inference", action="store_true")
    args = parser.parse_args()

    config_path = "configs"
    
    with initialize(version_base=None, config_path=config_path) as cfg:
        cfg = compose(config_name=args.config)
        if args.only_inference:
            cfg.do_training = False
            cfg.do_sampling = True
        start(cfg)