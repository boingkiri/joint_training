if __name__=="__main__":
    import sys
    sys.path.append("../")

import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from PIL import Image

from hydra import compose

from framework.diffusion.consistency_framework import CMFramework
from utils.common_utils import load_dataset_from_tfds
from utils.fs_utils import FSUtils

from tqdm import tqdm 

import wandb
import io
import os

# Calculate Pixel alignment error (PAE)

class PAEUtils():
    def __init__(self, consistency_config, wandb_obj=None) -> None:
        self.rng = jax.random.PRNGKey(42)

        self.n_timestep = 41
        self.num_denoiser_samples = consistency_config.sampling_batch # 256
        self.consistency_config = consistency_config
        self.num_consistency_samples_per_denoiser_sample = 32
        self.batchsize = 625

        sigma_min = 0.02
        sigma_max = 80
        rho = 7
        sweep_timestep = jnp.arange(self.n_timestep)
        self.t_steps = (sigma_max ** (1 / rho) + sweep_timestep / (self.n_timestep - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # self.sampled_data = self.sample_target_data()
        # self.calculate_and_save_ideal_denoiser()
        self.p_ideal_denoiser = jax.pmap(lambda x, t: self.calculate_ideal_denoiser_for_each_sample(x, t))

    def gather_all_data_in_dataset(self):
        def normalize_to_minus_one_to_one(image):
            return image * 2 - 1

        def normalize_channel_scale(image, label):
            image = tf.cast(image, tf.float32)
            image = (image / 255.0)
            image = normalize_to_minus_one_to_one(image)
            return image, label
        
        def augmentation(image, label):
            image, label = normalize_channel_scale(image, label)
            return image, label
        ds = tfds.load("cifar10", as_supervised=True)
        train_ds, _ = ds['train'], ds['test']
        train_ds = train_ds.map(augmentation)
        train_ds = train_ds.batch(self.batchsize)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        train_ds = map(lambda data: jax.tree_map(lambda x: x._numpy(), data), train_ds)
        return train_ds

    def sample_target_data(self):
        self.reset_dataset()
        data = next(self.datasets)
        data = data[0][:, 0, ...]
        return data

    def calculate_ideal_denoiser_for_each_sample(self, perturbed_sample, sigma):        
        gathered_dataset = self.gather_all_data_in_dataset()
        exp_component = jax.vmap(lambda y: -jnp.sum((perturbed_sample - y) ** 2, axis=(1, 2, 3)) / 2 / sigma ** 2)(next(gathered_dataset)[0])
        for (y, _) in gathered_dataset:
            exp_component = jnp.append(exp_component, jax.vmap(lambda y: -jnp.sum((perturbed_sample - y) ** 2, axis=(1, 2, 3)) / 2 / sigma ** 2)(y), axis=0)

        exp_component = exp_component - jax.scipy.special.logsumexp(exp_component, axis=0)
        coeff = jnp.exp(exp_component)

        gathered_dataset = self.gather_all_data_in_dataset()
        denoiser = jnp.zeros_like(perturbed_sample)
        for i, (y, _) in enumerate(gathered_dataset):
            denoiser += jax.vmap(lambda c: jnp.sum(c[:, None, None, None] * y, axis=0), in_axes=1)(coeff[self.batchsize*i : self.batchsize*(i+1), :])
        
        return denoiser

    def calculate_and_save_ideal_denoiser(self):
        p_calculate_ideal_denoiser_for_each_sample = jax.pmap(lambda x, sigma: self.calculate_ideal_denoiser_for_each_sample(x, sigma, self.rng))
        for sigma in tqdm(self.t_steps):
            perturbed_sample = self.sampled_data + jax.random.normal(self.rng, shape=self.sampled_data.shape) * sigma
            denoiser = p_calculate_ideal_denoiser_for_each_sample(perturbed_sample, jnp.repeat(sigma, jax.local_device_count()))
            # denoiser = self.calculate_ideal_denoiser_for_each_sample(self.sampled_data, sigma, self.rng)
            denoiser = jnp.reshape(denoiser, (-1, 32, 32, 3))
            self.save_img(denoiser, sigma)

    def reset_dataset(self):
        # Assume that the CIFAR10 would only be used  
        self.datasets = load_dataset_from_tfds(self.consistency_config, "cifar10", self.num_denoiser_samples, 1, x_flip=False, shuffle=False, for_pae=True)

    def save_img(self, samples, sigma):
        samples = (samples + 1) / 2
        samples = jnp.clip(samples, 0, 1)
        samples = jnp.asarray(samples * 255, dtype=jnp.uint8)

        os.makedirs(f'debug/{sigma}', exist_ok=True)
        for i, sample in enumerate(samples):
            Image.fromarray(np.asarray(sample)).save(f"debug/{sigma}/{i}.png")

    def viz_sample(self, sample):
        sample = jnp.asarray(sample)
        # sample = sample.reshape((-1, 32, 32, 3))
        # sample = sample[:2]
        
        sample = (sample + 1) / 2
        sample = jnp.clip(sample, 0, 1)
        sample = jnp.asarray(sample * 255, dtype=jnp.uint8)

        image_samples = []
        img_containter = np.zeros((32 * 2, 32, 3), dtype=np.uint8)
        for i in range(sample.shape[0]):
            img_set = sample[i]
            denoiser_img = img_set[0]
            consistency_img = img_set[1]
            img_containter[:32, :] = np.asarray(denoiser_img)
            img_containter[32:, :] = np.asarray(consistency_img)
            image_samples.append(img_containter)
            img_containter = np.zeros((32 * 2, 32, 3), dtype=np.uint8)

        image_samples = [Image.fromarray(np.asarray(image_samples[i])) for i in range(len(image_samples))]
        return image_samples

    def calculate_pae(self, consistency_framework: CMFramework, step: int):
        self.reset_dataset()
        data = next(self.datasets)
        data = data[0][:, 0, ...] # 256 samples (8 * 32) for TPU-v3, (4 * 64) for TPU-v4

        error_x_label = []
        error_y_label = []
        sample_images = []

        rng = self.rng

        print(f"Start to calculate for step {step}.")

        for timestep in range(0, self.n_timestep):
            print(f"Start {timestep} / {self.n_timestep}")
            
            rng, sampling_rng = jax.random.split(rng)
            noise = jax.random.normal(sampling_rng, shape=data.shape) * self.t_steps[timestep]

            # Get denoiser output
            denoiser_output = self.p_ideal_denoiser(data + noise, jnp.repeat(self.t_steps[timestep], jax.local_device_count()))

            # Get consistency output
            consistency_output = consistency_framework.sampling_cm_intermediate(self.num_denoiser_samples, original_data=data, sigma_scale=self.t_steps[timestep], noise=noise)
            consistency_output = jnp.reshape(consistency_output, data.shape)

            # Sample multiple datapoints
            sampling_list = []
            for i in tqdm(range(self.num_consistency_samples_per_denoiser_sample)): # 32
                rng, sampling_rng = jax.random.split(rng)

                new_noise = jax.random.normal(sampling_rng, shape=consistency_output.shape) * self.t_steps[timestep]
                second_consistency_output = consistency_framework.sampling_cm_intermediate(
                    self.num_denoiser_samples, original_data=consistency_output, sigma_scale=self.t_steps[timestep], noise=new_noise)
                sampling_list.append(second_consistency_output)

            sampling_list = jnp.stack(sampling_list, axis=0)
            sampling_list = jnp.mean(sampling_list, axis=0)
            denoiser_output = jnp.reshape(denoiser_output, (self.num_denoiser_samples, 32, 32, 3))
            pixel_alignment_error = jnp.mean(jnp.abs(sampling_list - denoiser_output), axis=(-1, -2, -3))

            sample_images.append([denoiser_output[0], sampling_list[0]])
            print("Total mean of error: ", jnp.mean(pixel_alignment_error))

            error_x_label.append(timestep)
            error_y_label.append(jnp.mean(pixel_alignment_error))

        total_pixel_alignment_error = jnp.mean(jnp.array(error_y_label))
        total_pixel_alignment_error_var = jnp.var(jnp.array(error_y_label))
        print(f"Total pixel alignment error: {total_pixel_alignment_error}")
        print(f"Total pixel alignment error variance: {total_pixel_alignment_error_var}")

        data = [[x, y] for (x, y) in zip(error_x_label, error_y_label)]
        wandb_table = wandb.Table(data=data, columns=['timestep', 'PAE'])
        wandb.log(
            {"Pixel Alignment error": wandb.plot.bar(
                wandb_table, "timestep", "PAE", title="Pixel Alignment error")
            },
            step=step
        )

        wandb.log({
            "train/Total Pixel Alignment error": total_pixel_alignment_error,
            "train/Total Pixel Alignment error variance": total_pixel_alignment_error_var
        }, step=step)

        np_image = self.viz_sample(sample_images)
        wandb_image = [wandb.Image(image, caption="Sample images") for image in np_image]
        wandb.log({"train/Sample images": wandb_image}, step=step)
        return total_pixel_alignment_error, total_pixel_alignment_error_var


# WARNING: This unit test is working only when the file is placed in the root directory 
# (in diffusion directory, not in utils directory)
if __name__=="__main__":
    from hydra import initialize
    from utils.log_utils import WandBLog
    if False:
        # consistency_config_path = "config_consistency"
        consistency_config_path = "config"

        args ={
                "project": "test",
                "name": "pae_unit_test",
            }
        wandb.init(**args)

        # with initialize(config_path="../configs") as cfg:
        with initialize(config_path="configs") as cfg:
            consistency_config = compose(config_name=consistency_config_path)
            consistency_config["do_training"] = False
            consistency_fs_obj = FSUtils(consistency_config)
            wandb_obj = WandBLog()
            pae_utils = PAEUtils(consistency_config, wandb_obj)

            framework_rng = jax.random.PRNGKey(42)
            consistency_framework = CMFramework(consistency_config, framework_rng, consistency_fs_obj, wandb_obj)
            mean, var = pae_utils.calculate_pae(consistency_framework, 0)
            print(mean, var)
    if True:
        consistency_config_path = "config"

        with initialize(config_path="../configs") as cfg:
        # with initialize(config_path="configs") as cfg:
            consistency_config = compose(config_name=consistency_config_path)
            consistency_config["do_training"] = False
            # consistency_fs_obj = FSUtils(consistency_config)
            pae_utils = PAEUtils(consistency_config)
            # breakpoint()
