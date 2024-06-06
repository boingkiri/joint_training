import wandb

class WandBLog():
    def __init__(self) -> None:
        self.log_dict = {}
    
    def update_log(self, log: dict):
        self.log_dict.update(log)
    
    def print_log(self):
        print(self.log_dict)
    
    def flush(self, step):
        wandb.log(self.log_dict, step=step)
        self.log_dict = {}
    