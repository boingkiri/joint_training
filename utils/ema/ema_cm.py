from utils.ema.default_ema import DefaultEMA 

import jax

class CMEMA(DefaultEMA):
    def __init__(
        self, 
        beta=0.9999, 
        update_every=1,
        update_after_step=1,
        is_distillation=False
        ):

        super().__init__(beta, update_every, update_after_step)

        def ema_update_pmap_fn(state, step=None):
            step = state.step if step is None else step
            current_decay = self.get_current_decay(step)
            ema_updated_params = jax.tree_map(
                lambda x, y: current_decay * x + (1 - current_decay) * y,
                state.params_ema, state.params)
            # state = state.replace(params_ema = jax.lax.stop_gradient(ema_updated_params))
            state = state.replace(params_ema = ema_updated_params)
            return state
        
        self.ema_update_pmap = jax.jit(ema_update_pmap_fn)

        self.is_distillation=is_distillation
        
    
    