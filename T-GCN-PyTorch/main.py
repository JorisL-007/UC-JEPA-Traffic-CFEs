import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl

import models
import tasks

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.data.metr_la_data import METRLADataModule


def main(args):
    pl.seed_everything(args.seed)
    dm = METRLADataModule(args.data, args.adj, args.batch_size, args.seq_len, args.pre_len)
    model = models.TGCN(dm.adj, args.hidden_dim, num_features=dm.num_features, pre_len=args.pre_len)
    task = tasks.SupervisedForecastTask(
        model=model,
        feat_max_val=dm.feat_max_val,
        loss=args.loss,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.output, monitor="val_loss")
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        default_root_dir=args.output,
        callbacks=[checkpoint],
    )
    trainer.fit(task, dm)
    return trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/metr_la_context.npz")
    parser.add_argument("--adj", type=Path, default=ROOT / "data/metr_la_adjacency.npz")
    parser.add_argument("--output", type=Path, default=Path("runs/tgcn"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pre_len", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--limit_train_batches", type=int)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser = models.TGCN.add_model_specific_arguments(parser)
    parser = tasks.SupervisedForecastTask.add_task_specific_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
