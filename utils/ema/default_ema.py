import jax
import jax.numpy as jnp

from flax.training.train_state import TrainState


class DefaultEMA():
    def __init__(
        self,
        beta=0.9999, 
        update_every=1,
        update_after_step=1,
        ):
        self.beta = beta
        self.update_every = update_every
        self.update_after_step = update_after_step

        def ema_update_pmap_fn(state):
            step = state.step
            current_decay = self.get_current_decay(step)
            ema_updated_params = jax.tree_map(
                lambda x, y: current_decay * x + (1 - current_decay) * y,
                state.params_ema, state.params)
            # state = state.replace(params_ema = jax.lax.stop_gradient(ema_updated_params))
            state = state.replace(params_ema = ema_updated_params)
            return state
        
        self.ema_update_pmap = jax.jit(ema_update_pmap_fn)
    
    def ema_update(self, state: TrainState, step: int=None):
        new_state = self.ema_update_pmap(state, step)
        return new_state

    def _clamp(self, value, min_value=None, max_value=None):
        # assert min_value is not None or max_value is not None
        if min_value is not None:
            value = jax.numpy.where(min_value > value, min_value, value)
        if max_value is not None:
            value = jax.numpy.where(max_value > value, value, max_value)
        return value


    def get_current_decay(self, step):
        effective_step = self._clamp(step - self.update_after_step - 1, min_value=0.)
        result_value = jax.numpy.where(effective_step <= 0, 0, self.beta)
        return result_value
    
    