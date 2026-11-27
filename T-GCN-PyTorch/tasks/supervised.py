import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import utils.losses


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss="mse",
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self._loss = loss
        self.feat_max_val = feat_max_val

    def forward(self, x):
        batch_size, _, num_nodes, _ = x.size()
        predictions = self.model(x)
        return predictions.reshape(batch_size, num_nodes, -1).transpose(1, 2)

    def shared_step(self, batch):
        mask = batch["y_mask"].bool()
        return self(batch["x"])[mask], batch["y"][mask]

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch)
        if y.numel() == 0:
            return None
        loss = self.loss(predictions, y)
        self.log("train_loss", loss, batch_size=y.numel())
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch)
        if y.numel() == 0:
            return
        loss = F.mse_loss(predictions, y) * self.feat_max_val ** 2
        self.log("val_loss", loss, batch_size=y.numel())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser
