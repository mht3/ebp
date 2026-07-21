'''
Wandb utils
'''

from typing import Optional

import wandb


class WandbLogger:
    """Thin wrapper around a wandb run."""

    def __init__(self, project_name: str = 'ebp', config: Optional[dict] = None) -> None:
        self.run = wandb.init(project=project_name, config=config)

    def update(self, metrics: dict, step: Optional[int] = None) -> None:
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        wandb.finish()
