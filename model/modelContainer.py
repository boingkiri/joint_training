import jax

class ModelContainer():
    def __init__(self, config: dict) -> None:
        self.model_type = config['model']
    
    