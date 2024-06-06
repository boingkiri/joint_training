from flax.training import train_state

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P

import optax
import flax.linen as nn
from flax.training import checkpoints, orbax_utils
import orbax.checkpoint

from omegaconf import DictConfig
from typing import Any

from functools import partial

class TrainState(train_state.TrainState):
  params_ema: Any = None

def get_framework_config(config: DictConfig, model_type):
  if model_type in ['autoencoder', 'discriminator']:
    framework_config = config.framework.autoencoder
  elif model_type in ['ldm']:
    if config['framework']['train_idx'] == 1:
      framework_config = config.framework.autoencoder
    elif config['framework']['train_idx'] == 2:
      framework_config = config.framework.diffusion
  else:
    framework_config = config.framework.diffusion
  return framework_config

def get_learning_rate_schedule(config: DictConfig, model_type, is_torso):
  tmp_config = get_framework_config(config, model_type)
  if is_torso is None:
    learning_rate = tmp_config['train']['learning_rate']
  elif is_torso is False:
    learning_rate = tmp_config['train']['head_learning_rate']
  elif is_torso is True:
    learning_rate = tmp_config['train']['torso_learning_rate']
  if "warmup" in tmp_config['train']:
    learning_rate = optax.warmup_exponential_decay_schedule(
      init_value=0.0,
      peak_value=learning_rate,
      warmup_steps=tmp_config['train']['warmup'],
      decay_rate=1,
      transition_steps=1
    )
  else:
    learning_rate = optax.constant_schedule(learning_rate)
  return learning_rate


def create_optimizer(config: DictConfig, model_type, is_torso=None):
  # Initialize the optimizer
  learning_rate = get_learning_rate_schedule(config, model_type, is_torso)
  framework_config = get_framework_config(config, model_type)
  # Gradient Clipping
  optax_chain = []
  if "gradient_clip" in framework_config['train']:
    optax_chain.append(optax.clip(framework_config['train']['gradient_clip']))
  optimizer_config = framework_config['train']['optimizer']

  # Setting Optimizer
  if optimizer_config['type'] == "Adam":
    betas = [0.9, 0.999] if "betas" not in optimizer_config else optimizer_config['betas']
    optax_chain.append(optax.adam(learning_rate, b1=betas[0], b2=betas[1]))
  elif optimizer_config['type'] == "radam":
    optax_chain.append(optax.radam(learning_rate))
  tx = optax.chain(*optax_chain)

  # Create gradient accumulation
  # if framework_config['train'].get("batch_size_per_rounds", None) is not None and \
  #     framework_config['train']["total_batch_size"] != framework_config['train']["batch_size_per_rounds"]:
  # if framework_config['train'].get("batch_size_per_rounds", None) is not None:
  if framework_config['train'].get("batch_size_per_rounds", None) is not None and framework_config.get("type", "diffusion") != "edm": # TMP: need to fix TODO
    assert framework_config['train']["total_batch_size"] % framework_config['train']["batch_size_per_rounds"] == 0
    num_of_rounds = framework_config['train']["total_batch_size"] // framework_config['train']["batch_size_per_rounds"]
    tx = optax.MultiSteps(tx, every_k_schedule=num_of_rounds)
  return tx


# @partial(jax.pmap, static_broadcasted_argnums=(0, 1, 2, 4))
# def create_train_state(config: DictConfig, model_type, model, rng, aux_data=None):
def create_train_state(config: DictConfig, model_type, apply_fn, params):
  """
  Creates initial 'TrainState'
  """
  tx = create_optimizer(config, model_type)

  # Return the training state
  return TrainState.create(
      apply_fn=apply_fn,
      params=params,
      params_ema=params,
      tx=tx
  )

def save_train_state(state, checkpoint_dir, step, prefix=None):
  if prefix is None:
    prefix = "checkpoint_"
  # checkpoints.save_checkpoint(checkpoint_dir, state, step, prefix=prefix)

  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(state)
  orbax_checkpointer.save(checkpoint_dir, state, )
  print(f"Saving {step} complete.")


def load_state_from_checkpoint_dir(checkpoint_dir, state, step, checkpoint_prefix="checkpoint_"):
    state = checkpoints.restore_checkpoint(checkpoint_dir, state, prefix=checkpoint_prefix, step=step)
    if type(state) is dict:
      print(f"Checkpoint {state['step']} loaded")
    else:
      print(f"Checkpoint {state.step} loaded")
    return state

def save_best_state(state, best_checkpoint_dir, step, checkpoint_prefix):
  assert type(state) is dict
  for key in state:
    checkpoints.save_checkpoint(best_checkpoint_dir, state[key], step, prefix=key + "_", overwrite=True)
  print(f"Best {step} steps! Saving {step} in best checkpoint dir complete.")

def create_environment_sharding(config: DictConfig):
  """Creates a sharding configuration for the environment."""
  model_parallelism = config.get("model_parallel_device", 1)

  # devices = np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count())
  devices = np.array(jax.devices()).reshape(jax.device_count() // model_parallelism, model_parallelism)
  axes_names = ('data_parallelism', 'model_parallelism')
  global_mesh = jax.sharding.Mesh(devices, axes_names)
  return global_mesh, axes_names

def unreplicate_tree(tree):
  """Returns a single instance of a replicated array."""
  return jax.tree_util.tree_map(lambda x: x[0][0], tree)

def modified_fully_replicated_host_local_array_to_global_array(
    arr: jax.Array,
) -> jax.Array:
  """Converts a host local array from to global jax.Array.

  In most cases, the local array is expected to have been produced by pmap.

  Args:
    arr: Host local array

  Returns:
    A global array.
  """
  # if not arr.is_fully_replicated:
  #   raise ValueError('Array must be fully replicated.')
  global_shape = arr.addressable_data(0).shape
  # Create a 1D mesh to create fully replicated global jax.Array.
  sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',)),
      jax.sharding.PartitionSpec(None)
      if global_shape
      else jax.sharding.PartitionSpec(),
  )
  # pmap-produced Array has a "scrambled" device order.
  dbs = sorted(
      [shard.data for shard in arr.addressable_shards],
      key=lambda x: x.device().id,
  )
  return jax.make_array_from_single_device_arrays(global_shape, sharding, dbs)