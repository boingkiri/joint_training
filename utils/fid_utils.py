import jax
import jax.numpy as jnp
import numpy as np

from utils.fid import inception, fid
from utils.fs_utils import FSUtils

import functools
import os
import shutil

from omegaconf import DictConfig

class FIDUtils():
    def __init__(self, config: DictConfig) -> None:
        self.rng = jax.random.PRNGKey(42)
        self.img_size = (299, 299)
        self.model, self.params, self.apply_fn = self.load_fid_model()
        
        self.fs_utils = FSUtils(config)
        self.in_process_dir = config.exp.in_process_dir
        self.dataset_name = config.dataset.name
    
    def load_fid_model(self):
        model = inception.InceptionV3(pretrained=True)
        params = model.init(self.rng, jnp.ones((1, 256, 256, 3)))
        apply_fn = jax.jit(functools.partial(model.apply, train=False))
        return model, params, apply_fn
    
    def get_tmp_dir(self):
        tmp_dir = os.path.join(self.in_process_dir, "tmp")
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir) 
        return tmp_dir
    
    def precompute_dataset(self, dataset_name):
        if dataset_name == "cifar10" and "stats.npz" not in os.listdir(dataset_name):
            print("Precomputing CIFAR10 statistics")
            dataset_path = os.path.join(dataset_name, "train")
            mu, sigma = fid.compute_statistics(dataset_path, self.params, self.apply_fn, 50, self.img_size)
            statistics_file = os.path.join(dataset_name, "stats")
            np.savez(statistics_file, mu=mu, sigma=sigma)
            return mu, sigma
        print(f"Loading {dataset_name} statistics")
        statistics_file = os.path.join(dataset_name, "stats.npz")
        mu, sigma = fid.compute_statistics(statistics_file, self.params, self.apply_fn, 50, self.img_size)
        return mu, sigma

    def calculate_statistics(self, img_path):
        mu, sigma = fid.compute_statistics(img_path, self.params, self.apply_fn, 50, self.img_size)
        return mu, sigma

    def calculate_fid(self, src_img_path, des_img_path=None):
        if des_img_path is None:
            dest_mu, dest_sigma = self.precompute_dataset(self.dataset_name)
        else:
            dest_mu, dest_sigma = self.calculate_statistics(des_img_path)
        src_mu, src_sigma = self.calculate_statistics(src_img_path)
        fid_score = fid.compute_frechet_distance(src_mu, dest_mu, src_sigma, dest_sigma)
        return fid_score
    
    def calculate_fid_and_mu_diff(self, src_img_path, des_img_path=None):
        if des_img_path is None:
            dest_mu, dest_sigma = self.precompute_dataset(self.dataset_name)
        else:
            dest_mu, dest_sigma = self.calculate_statistics(des_img_path)
        src_mu, src_sigma = self.calculate_statistics(src_img_path)
        fid_score = fid.compute_frechet_distance(src_mu, dest_mu, src_sigma, dest_sigma)
        mu_diff = fid.compute_mean_difference(src_mu, dest_mu)
        return fid_score, mu_diff
    
    def save_images_for_fid(self, model_obj, total_num_samples, batch_size, sampling_mode=None):
        tmp_dir = self.get_tmp_dir()
        tmp_dir = os.path.join(self.in_process_dir, "tmp")

        current_num_samples = 0
        random_key = jax.random.PRNGKey(42)
        while current_num_samples < total_num_samples:
            random_key, sampling_key = jax.random.split(random_key)
            if sampling_mode is not None:
                sample = model_obj.sampling(batch_size, mode=sampling_mode, random_key=sampling_key)
            else:
                sample = model_obj.sampling(batch_size, random_key=sampling_key)
            sample = jnp.reshape(sample, (batch_size, *sample.shape[-3:]))
            current_num_samples += self.fs_utils.save_images_to_dir(sample, tmp_dir, current_num_samples)
        
        remained_samples = current_num_samples - total_num_samples
        if remained_samples > 0:
            self.fs_utils.delete_images_from_dir(tmp_dir, total_num_samples)
            current_num_samples -= remained_samples
        return tmp_dir

    def calculate_fid_in_step(self, model_obj, total_num_samples, batch_size=128, sampling_mode=None):
        tmp_dir = self.save_images_for_fid(model_obj, total_num_samples, batch_size, sampling_mode)
        # fid_score, mu_diff = self.calculate_fid(tmp_dir)
        fid_score, mu_diff = self.calculate_fid_and_mu_diff(tmp_dir)
        shutil.rmtree(tmp_dir)
        return fid_score, mu_diff
    
    def print_and_save_fid(self, step, fid_score, sampling_mode=None, mu_diff=None):
        writing_format = f"FID score of Step {step} : {fid_score:.4f}\n" if fid_score is not None else f"FID score of Step {step} : None\n"
        if mu_diff is not None: 
            writing_format += f"Mean difference of Step {step} : {mu_diff:.4f}\n" if fid_score is not None else f"Mean difference of Step {step} : None\n"
        print(writing_format)

        file_name = f"fid_log_{sampling_mode}.txt" if sampling_mode is not None else "fid_log.txt" 
        fid_log_file = os.path.join(self.in_process_dir, file_name)
        with open(fid_log_file, 'a') as f:
            f.write(writing_format)

