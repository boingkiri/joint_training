
import os
import shutil
import yaml
import io

from . import common_utils, jax_utils
from omegaconf import DictConfig, OmegaConf

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import jax
import flax
import orbax.checkpoint

class FSUtils():
    def __init__(self, config: DictConfig) -> None:
        self.config = config

        # Create pytree checkpointer and its manager
        # self.checkpoint_manager, self.best_checkpoint_manager = self.create_checkpoint_manager() # orbax.checkpoint.CheckpointManager. Dict[CheckpointManager]

        self.checkpoint_manager, self.best_checkpoint_manager, self.tmp_checkpoint_manager = self.create_checkpoint_manager() # orbax.checkpoint.CheckpointManager. Dict[CheckpointManager]

    def create_checkpoint_manager(self):
        model_keys = self.config.model.keys()
        best_checkpoint_manager = {}
        abs_path_ = os.getcwd()+"/"
        # model_checkpoint_manager_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, single_host_load_and_broadcast=True)
        model_checkpoint_manager_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1)
        self.verify_and_create_dir(abs_path_ + self.config.exp.checkpoint_dir)
        model_checkpoint_manager = orbax.checkpoint.CheckpointManager(
            abs_path_ + self.config.exp.checkpoint_dir, 
            item_names=model_keys,
            options=model_checkpoint_manager_options,
            item_handlers={
                model_key: orbax.checkpoint.StandardCheckpointHandler(primary_host=None) for model_key in model_keys
            },
            primary_host=None,
        )
        # TMP: to save checkpoint at 200k or 300k
        # tmp_checkpoint_manager_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, single_host_load_and_broadcast=True)
        tmp_checkpoint_manager_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2)
        self.verify_and_create_dir(abs_path_ + self.config.exp.checkpoint_dir + "/tmp")
        tmp_checkpoint_manager = orbax.checkpoint.CheckpointManager(
            abs_path_ + self.config.exp.checkpoint_dir + "/tmp", 
            item_names=model_keys,
            options=tmp_checkpoint_manager_options,
            item_handlers={
                model_key: orbax.checkpoint.StandardCheckpointHandler(primary_host=None) for model_key in model_keys
            },
            primary_host=None,
        )

        for model_key in model_keys:
            best_checkpoint_dir = self.config.exp.best_dir + "/" + model_key
            model_best_checkpoint_manager_options = orbax.checkpoint.CheckpointManagerOptions(
                # max_to_keep=1, best_fn=lambda metrics: metrics['fid'], best_mode='min', single_host_load_and_broadcast=True)
                max_to_keep=1, best_fn=lambda metrics: metrics['fid'], best_mode='min')
            self.verify_and_create_dir(abs_path_ + best_checkpoint_dir)
            model_best_checkpoint_manager = orbax.checkpoint.CheckpointManager(
                abs_path_ + best_checkpoint_dir,
                item_names=model_keys,
                options=model_best_checkpoint_manager_options,
                item_handlers={
                    model_key: orbax.checkpoint.StandardCheckpointHandler(primary_host=None) for model_key in model_keys
                },
                primary_host=None,
            )
            best_checkpoint_manager[model_key] = model_best_checkpoint_manager
        

        return model_checkpoint_manager, best_checkpoint_manager, tmp_checkpoint_manager
    
    def update_checkpoint_manager(self, add_model_keys=None, delete_model_keys=None):
        model_keys = set(self.config.model.keys())
        best_checkpoint_manager = {}
        abs_path_ = os.getcwd()+"/"

        if add_model_keys is not None:
            for add_model_key in add_model_keys:
                model_keys.add(add_model_key)
        if delete_model_keys is not None:
            for delete_model_key in delete_model_keys:
                model_keys.discard(delete_model_key)

        # model_checkpoint_manager_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, single_host_load_and_broadcast=True)
        model_checkpoint_manager_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1)
        self.verify_and_create_dir(abs_path_ + self.config.exp.checkpoint_dir)
        model_checkpoint_manager = orbax.checkpoint.CheckpointManager(
            abs_path_ + self.config.exp.checkpoint_dir, 
            # checkpointers={model_key: orbax.checkpoint.PyTreeCheckpointer() for model_key in model_keys},
            options=model_checkpoint_manager_options,
            item_handlers={
                model_key: orbax.checkpoint.StandardCheckpointHandler(primary_host=None) for model_key in model_keys
            },
            primary_host=None,
            )
            # item_handlers={model_key: orbax.checkpoint.StandardCheckpointHandler() for model_key in model_keys})

        for model_key in model_keys:
            best_checkpoint_dir = self.config.exp.best_dir + "/" + model_key
            model_best_checkpoint_manager_options = orbax.checkpoint.CheckpointManagerOptions(
                # max_to_keep=1, best_fn=lambda metrics: metrics['fid'], best_mode='min', single_host_load_and_broadcast=True)
                max_to_keep=1, best_fn=lambda metrics: metrics['fid'], best_mode='min')
            self.verify_and_create_dir(abs_path_ + best_checkpoint_dir)
            model_best_checkpoint_manager = orbax.checkpoint.CheckpointManager(
                abs_path_ + best_checkpoint_dir,
                # checkpointers={model_key: orbax.checkpoint.PyTreeCheckpointer() for model_key in model_keys}, 
                options=model_best_checkpoint_manager_options,
                item_handlers={
                    model_key: orbax.checkpoint.StandardCheckpointHandler(primary_host=None) for model_key in model_keys
                },
                primary_host=None,
                )
                # item_handlers={model_key: orbax.checkpoint.StandardCheckpointHandler() for model_key in model_keys})
            best_checkpoint_manager[model_key] = model_best_checkpoint_manager
        self.checkpoint_manager = model_checkpoint_manager
        self.best_checkpoint_manager = best_checkpoint_manager

    def verify_and_create_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"{dir_path} created.")

    def verify_and_create_workspace(self):
        # Creating current exp dir
        current_exp_dir = self.config.exp.current_exp_dir
        self.verify_and_create_dir(current_exp_dir)
        
        if self.config["do_training"]:
            # Creating config file
            config_filepath = os.path.join(current_exp_dir, 'config.yaml')
            with open(config_filepath, 'w') as f:
                yaml.dump(OmegaConf.to_container(self.config, resolve=True), f)
        
            # Copying python file to current exp dir
            python_filepath = os.path.join(current_exp_dir, 'python_files')
            workspace_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
            for walking_path in os.walk(workspace_path):
                files = walking_path[2]
                walking_path = walking_path[0]
                walking_rel_path = os.path.relpath(walking_path, workspace_path)
                saving_filepath = os.path.join(python_filepath, walking_rel_path)
                if self.config.exp.exp_dir in walking_rel_path:
                    continue
                if "bin" in walking_rel_path or "include" in walking_rel_path or "lib" in walking_rel_path or "share" in walking_rel_path:
                    continue
                elif os.path.isdir(walking_path) and not os.path.exists(saving_filepath):
                    os.makedirs(saving_filepath)
                for file in files:
                    if ".py" in file and not ".pyc" in file:
                        shutil.copy(os.path.join(walking_path, file), saving_filepath)
        
        # Creating checkpoint dir
        checkpoint_dir = self.config.exp.checkpoint_dir
        self.verify_and_create_dir(checkpoint_dir)
        
        # Creating best checkpoint dir
        best_checkpoint_dir = self.config.exp.best_dir
        self.verify_and_create_dir(best_checkpoint_dir)
        
        # Creating sampling dir
        sampling_dir = self.config.exp.sampling_dir
        self.verify_and_create_dir(sampling_dir)
        
        # Creating in_process dir
        in_process_dir = self.config.exp.in_process_dir
        self.verify_and_create_dir(in_process_dir)
        
        # Creating dataset dir
        dataset_name = self.config.dataset.name
        if not os.path.exists(dataset_name):
            print("Creating dataset dir")
            os.makedirs(dataset_name)
            if dataset_name == "cifar10":
                import tarfile
                common_utils.download("http://pjreddie.com/media/files/cifar.tgz", dataset_name)
                filepath = os.path.join(dataset_name, "cifar.tgz")
                file = tarfile.open(filepath)
                file.extractall(dataset_name)
                file.close()
                os.remove(filepath)
                train_dir_path = os.path.join(dataset_name, "cifar", "train")
                dest_dir_path = os.path.join(dataset_name, "train")
                os.rename(train_dir_path, dest_dir_path)

    def get_start_step_from_checkpoint(self):
        latest_step = self.checkpoint_manager.latest_step()
        if latest_step is None:
            return 0
        return latest_step
    
    def make_image_grid(self, images):
        images = common_utils.unnormalize_minus_one_to_one(images)
        n_images = len(images)
        f, axes = plt.subplots(n_images // 4, 4)
        images = np.clip(images, 0, 1)
        axes = np.concatenate(axes)

        for img, axis in zip(images, axes):
            axis.imshow(img)
            axis.axis('off')
        return f
    
    def get_pil_from_np(self, images):
        f = self.make_image_grid(images)
        img_buf = io.BytesIO()
        f.savefig(img_buf, format='png')

        im = Image.open(img_buf)
        plt.close()
        return im
        # return Image.frombytes('RGB', f.canvas.get_width_height(),f.canvas.tostring_rgb())

    def save_comparison(self, images, steps, savepath):
        # Make in process dir first
        self.verify_and_create_dir(savepath)

        f = self.make_image_grid(images)
        
        save_filename = os.path.join(savepath, f"{steps}.png")
        f.savefig(save_filename)
        plt.close()
        return save_filename
    
    def save_images_to_dir(self, images, save_path_dir=None, starting_pos=0):
        current_sampling = 0
        if save_path_dir is None:
            save_path_dir = self.config.exp.sampling_dir
        images = common_utils.unnormalize_minus_one_to_one(images)
        images = np.clip(images, 0, 1)
        images = images * 255
        images = np.array(images).astype(np.uint8)
        for image in images:
            im = Image.fromarray(image)
            sample_path = os.path.join(save_path_dir, f"{starting_pos + current_sampling}.png")
            im.save(sample_path)
            current_sampling += 1
        return current_sampling
    
    def delete_images_from_dir(self, save_path_dir=None, starting_pos=50000):
        if save_path_dir is None:
            save_path_dir = self.config.exp.sampling_dir
        for content in os.listdir(save_path_dir):
            number = int(content.split(".")[0])
            if number >= starting_pos:
                os.remove(os.path.join(save_path_dir, content))

    def save_tmp_model_state(self, states, step):
        self.tmp_checkpoint_manager.save(step, states)
        print(f"TMP SAVE: Saving {step} complete.")

    def save_model_state(self, states, step, metrics=None):
        print(f"Get into the save_model_state")
        best_saved = False
        if self.config.get("distributed_training", False):
            # states = jax.tree_map(lambda x: jax.experimental.multihost_utils.broadcast_one_to_all(x), states)
            states = flax.jax_utils.replicate(states)
            states = jax.tree_map(lambda x: jax_utils.modified_fully_replicated_host_local_array_to_global_array(x), states)
        self.checkpoint_manager.save(
            step, 
            args=orbax.checkpoint.args.Composite(
                **{
                    k: orbax.checkpoint.args.StandardSave(v)
                    for k, v in states.items() if v is not None
                }
            )
        )
        self.checkpoint_manager.wait_until_finished()
        for state in states:
            best_checkpoint_manager = self.best_checkpoint_manager[state]
            # state_saved = best_checkpoint_manager.save(step, states, metrics=metrics[state])
            state_saved = best_checkpoint_manager.save(
                step, 
                metrics=metrics[state],
                args=orbax.checkpoint.args.Composite(
                    **{
                        k: orbax.checkpoint.args.StandardSave(v)
                        for k, v in states.items() if v is not None
                    }
                )
            )
            best_saved = best_saved or state_saved
            best_checkpoint_manager.wait_until_finished()
        
        print(f"Saving {step} complete.")
        if best_saved:
            print(f"Best {step} steps! Saving {step} in best checkpoint dir complete.")

    def load_model_state(self, state):
        print(f"Get into the load_model_state")
        step = self.checkpoint_manager.latest_step()
        # state = flax.jax_utils.replicate(state)
        # state = jax.tree_map(lambda x: jax_utils.modified_fully_replicated_host_local_array_to_global_array(x), state)
        if step is not None:
            abstract_train_state = jax.tree_util.tree_map(
                orbax.checkpoint.utils.to_shape_dtype_struct, state
            )
            state = self.checkpoint_manager.restore(
                step, 
                args=orbax.checkpoint.args.Composite(
                    **{
                        k: orbax.checkpoint.args.StandardRestore(v)
                        for k, v in abstract_train_state.items() if v is not None
                    }
                )
            )
            print(f"Loading ckpt of Step {step} complete.")
        else:
            print("No ckpt loaded. Start from scratch.")
        # if self.config.get("distributed_training", False):
        #     # state = jax.tree_map(lambda x: jax.experimental.multihost_utils.broadcast_one_to_all(x), state)
        #     state = flax.jax_utils.replicate(state)
        #     state = jax.tree_map(lambda x: jax_utils.modified_fully_replicated_host_local_array_to_global_array(x), state)
        #     print("Loading ckpt complete.")
        return state
    
    def get_best_fid(self):
        best_fid = {}
        in_process_dir = self.config.exp.in_process_dir

        for content in os.listdir(in_process_dir):
            if content.startswith("fid_log"):
                fid_log_key = content.split("_")[-1]
                fid_log_file = os.path.join(in_process_dir, content)
                with open(fid_log_file, 'r') as f:
                    txt = f.read()
                    logs = txt.split('\n')
                    for log in logs:
                        if len(log) == 0:
                            continue
                        frag = log.split(' ')
                        value = float(frag[-1])
                        if best_fid is None:
                            best_fid[fid_log_key] = value
                        elif best_fid[fid_log_key] >= value:
                            best_fid[fid_log_key] = value
        return best_fid

