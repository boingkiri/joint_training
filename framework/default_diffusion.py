from typing import TypedDict

from abc import *
from omegaconf import DictConfig

class DefaultModel(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, x, cond=None):
        pass
    
    @abstractmethod
    def sampling(self, num_image, img_size=None):
        pass

    @abstractmethod
    def init_model_state(self, config: DictConfig, model_type, model, rng):
        pass

    @abstractmethod
    def get_model_state(self):
        pass

    

    