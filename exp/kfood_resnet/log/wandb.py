from . import Logger
import os

class WandbLogger(Logger):
    def __init__(self, entity, api_key, group, name, project, tags, description=None, scripts=None, args=None):
        os.system(f"wandb login {api_key}")
        import wandb
        
        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            tags = tags,
            )
        wandb.run.name = name
        wandb.config.update(args)
        wandb.run.save()

    def log_metric(self, name, value, step=1):
        self.wandb.log(
            {name:value,
                'step':step}
        )

    def log_text(self, name, text):
        self.wandb.log({name: text})
        
    def log_image(self, name, image):
        self.wandb.log({name: self.wandb.Image(image)})

    def log_arguments(self, dictionary):
        pass
    
    def save_model(self, name, state_dict):
        pass

    def finish(self):
        self.wandb.finish()