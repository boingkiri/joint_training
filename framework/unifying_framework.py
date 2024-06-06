from framework.diffusion.consistency_framework import CMFramework

import jax
import jax.numpy as jnp

from utils.fs_utils import FSUtils
from utils import jax_utils, common_utils
from utils.fid_utils import FIDUtils
from utils.log_utils import WandBLog
from utils.pae_utils import PAEUtils

from tqdm import tqdm
import os
import wandb

from omegaconf import DictConfig

class UnifyingFramework():
    """
        This framework contains overall methods for training and sampling
    """
    def __init__(self, model_type, config: DictConfig, random_rng) -> None:
        self.config = config
        self.current_model_type = model_type.lower()
        self.random_rng = random_rng
        self.dataset_name = config.dataset.name
        self.do_fid_during_training = config.fid_during_training
        self.n_jitted_steps = config.get("n_jitted_steps", 1)
        self.dataset_x_flip = self.config.framework.diffusion.get("augment_rate", None) is None
        self.set_train_step_process(config)
    
    def set_train_step_process(self, config: DictConfig):
        self.set_utils(config)
        self.set_model(config)
        self.set_step(config)
        # self.learning_rate_schedule = jax_utils.get_learning_rate_schedule(config, self.current_model_type)
        self.learning_rate_schedule = jax_utils.get_learning_rate_schedule(config, self.current_model_type, is_torso=True)
        self.sample_batch_size = config.sampling_batch
    
    def set_utils(self, config: DictConfig):
        self.fid_utils = FIDUtils(config)
        self.fs_utils = FSUtils(config)
        self.wandblog = WandBLog()
        self.fs_utils.verify_and_create_workspace()

        # TMP
        if config["PAE_evaluation"]:
            self.pae_utils = PAEUtils(config, self.wandblog)

    def set_model(self, config: DictConfig):
        if self.current_model_type in ['cm', 'cm_diffusion']:
            diffusion_rng, self.random_rng = jax.random.split(self.random_rng, 2)
            self.framework = CMFramework(config, diffusion_rng, self.fs_utils, self.wandblog)
        else:
            NotImplementedError("Model Type cannot be identified. Please check model name.")
        
    def set_step(self, config: DictConfig):
        self.step = self.fs_utils.get_start_step_from_checkpoint()
        if self.current_model_type == "ldm":
            self.train_idx = config['framework']['train_idx']
            if self.train_idx == 1: # AE
                self.total_step = config['framework']['autoencoder']['train']['total_step']
                self.checkpoint_prefix = config.exp.autoencoder_prefix
            elif self.train_idx == 2: # Diffusion
                self.total_step = config['framework']['diffusion']['train']['total_step']
                self.checkpoint_prefix = config.exp.diffusion_prefix
        else:
            self.total_step = config['framework']['diffusion']['train']['total_step']
            self.checkpoint_prefix = config.exp.diffusion_prefix

    def sampling(self, num_img, original_data=None, mode=None):
        if mode is None:
            sample = self.framework.sampling(num_img, original_data=original_data)
        else:
            sample = self.framework.sampling(num_img, original_data=original_data, mode=mode)
        sample = jnp.reshape(sample, (num_img, *sample.shape[-3:]))
        return sample
    
    def save_model_state(self, states:dict, metrics:dict=None):
        self.fs_utils.save_model_state(states, self.step, metrics=metrics)

    def train(self):
        # TODO: The connection_denoiser_type is only used in CM training. need to be fixed.
        datasets = common_utils.load_dataset_from_tfds(self.config)
        datasets_bar = tqdm(datasets, total=self.total_step, initial=self.step)
        in_process_dir = self.config.exp.in_process_dir
        in_process_model_dir_name = 'diffusion'
        in_process_dir = os.path.join(in_process_dir, in_process_model_dir_name)
        num_of_rounds = self.config["framework"]["diffusion"]['train']["total_batch_size"] // \
                            self.config["framework"]["diffusion"]['train']["batch_size_per_rounds"]
        # first_step = True
        first_step = False
        fid_dict = {}
        log = {}

        num_used_dataset = 0

        for x, _ in datasets_bar:
            training_log = self.framework.fit(x, step=self.step)
            log.update(training_log)

            if self.step % self.config["sampling_step"] == 0:
                batch_data = x[0, 0, :8] # (device_idx, n_jitted_steps, batch_size)

                # Change of the sample quality is tracked to know how much the CM model is corrupted.
                # Sample generated image for EDM
                sample = self.sampling(self.sample_batch_size, original_data=batch_data, mode="edm")
                edm_xset = jnp.concatenate([sample[:8], batch_data], axis=0)
                sample_image = self.fs_utils.get_pil_from_np(edm_xset)
                # sample_path = self.fs_utils.save_comparison(edm_xset, self.step, in_process_dir)
                log['Sampling'] = wandb.Image(sample_image, caption=f"Step: {self.step}")

                # Sample generated image for training CM
                sample = self.sampling(self.sample_batch_size, original_data=batch_data, mode="cm-training")
                training_cm_xset = jnp.concatenate([sample[:8], batch_data], axis=0)
                sample_image = self.fs_utils.get_pil_from_np(training_cm_xset)
                log['Training CM Sampling'] = wandb.Image(sample_image, caption=f"Step: {self.step}")
                sample_path = self.fs_utils.save_comparison(training_cm_xset, self.step, in_process_dir) # TMP
                sample_image.close()
                
                self.wandblog.update_log(log)
                self.wandblog.flush(step=self.step)

            if self.step % self.config["saving_step"] == 0 and self.step not in fid_dict:
                model_state = self.framework.get_model_state()
                # sampling_modes = ['one-step', 'two-step', 'edm']
                sampling_modes = ['one-step', 'two-step']
                
                if self.do_fid_during_training and not (self.current_model_type == "ldm" and self.train_idx == 1):
                    mode_metrics = {}
                    for mode in sampling_modes:
                        fid_score, mu_diff = self.fid_utils.calculate_fid_in_step(self.framework, 50000, batch_size=self.sample_batch_size, sampling_mode=mode)
                        self.fid_utils.print_and_save_fid(self.step, fid_score, sampling_mode=mode, mu_diff=mu_diff)
                        metrics = {"fid": fid_score}
                        if mode == "edm":
                            mode_metrics["head"] = metrics
                            self.wandblog.update_log({"Head FID score": fid_score})
                        if mode == "two-step":
                            self.wandblog.update_log({"Two step FID score": fid_score})
                        elif mode == "one-step":
                            mode_metrics['diffusion'] = metrics
                            self.wandblog.update_log({"One step FID score": fid_score})
                        else:
                            NotImplementedError("Sampling mode is not implemented.")
                    if not first_step:
                        self.save_model_state(model_state, mode_metrics)
                    self.wandblog.flush(step=self.step)
                    fid_dict[self.step] = mode_metrics
                if self.config["PAE_evaluation"]:
                    self.pae_utils.calculate_pae(self.framework, self.step)

            
            description_str = "Step: {step}/{total_step} lr*1e4: {lr:.4f} ".format(
                step=self.step,
                total_step=self.total_step,
                lr=self.learning_rate_schedule(self.step)*(1e+4)
            )
            
            for key in log:
                if key.startswith("train"):
                    represented_key = key.replace("train/", "")
                    description_str += f"{represented_key}: {log[key]:.4f} "
            datasets_bar.set_description(description_str)

            if self.step >= self.total_step:
                if not self.next_step():
                    break
            num_used_dataset += self.n_jitted_steps
            
            update_step = num_used_dataset // num_of_rounds
            num_used_dataset = num_used_dataset % num_of_rounds
            
            datasets_bar.update(update_step)
            self.step += update_step
            first_step = False
            

    def sampling_and_save(self, total_num, img_size=None, mode="one-step"):
        if img_size is None:
            img_size = common_utils.get_dataset_size(self.dataset_name)
        current_num = 0
        batch_size = self.sample_batch_size
        while total_num > current_num:
            samples = self.sampling(batch_size, original_data=None, mode=mode)
            self.fs_utils.save_images_to_dir(samples, starting_pos=current_num)
            current_num += batch_size
        self.fs_utils.delete_images_from_dir(starting_pos=total_num)
    
    def reconstruction(self, total_num):
        img_size = common_utils.get_dataset_size(self.dataset_name)
        datasets = common_utils.load_dataset_from_tfds(n_jitted_step=self.n_jitted_steps)
        datasets_bar = tqdm(datasets, total=total_num)
        current_num = 0
        for x, _ in datasets_bar:
            batch_size = x.shape[0]
            samples = self.sampling(batch_size, img_size, original_data=x)
            self.fs_utils.save_images_to_dir(samples, starting_pos = current_num)
            current_num += batch_size
            if current_num >= total_num:
                break
    
    def next_step(self):
        if self.current_model_type == "ldm" and self.train_idx == 1:
            self.config['framework']['train_idx'] = 2
            self.set_train_step_process(self.config)
            return True
        else:
            return False

