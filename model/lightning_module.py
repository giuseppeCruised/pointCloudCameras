import pytorch_lightning as pl
import torch

from ReconstructionModule import ReconstructionModule
from util import chamfer_l1_safe


class MultiViewReconstructionModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ReconstructionModule()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        rec = self(x)
        loss = chamfer_l1_safe(x, rec)
        self.log("train_loss", loss)
       return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_training_epoch_end(self):
        batch = next(iter(self.trainer.datamodule.val_dataloader()))

