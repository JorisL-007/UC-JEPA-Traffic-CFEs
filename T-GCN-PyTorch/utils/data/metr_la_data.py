import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import METRLAContextDataset


class METRLADataModule(pl.LightningDataModule):
    def __init__(self, data_path, adj_path, batch_size=32, seq_len=12, pre_len=12):
        super().__init__()
        self.batch_size = batch_size
        with np.load(data_path, allow_pickle=False) as data:
            speeds = data["speeds_kmh"].astype(np.float32)
            dynamic = data["dynamic"].astype(np.float32)
            static = data["static"].astype(np.float32)
            self.sensor_ids = data["sensor_ids"].astype(str)

        with np.load(adj_path, allow_pickle=False) as graph:
            graph_ids = {sensor: i for i, sensor in enumerate(graph["sensor_ids"].astype(str))}
            try:
                order = [graph_ids[sensor] for sensor in self.sensor_ids]
            except KeyError as error:
                raise ValueError(f"Missing sensor in graph: {error.args[0]}") from error
            self.adj = graph["adjacency"][np.ix_(order, order)].astype(np.float32)
        np.fill_diagonal(self.adj, 0)  # T-GCN adds the self-loops.

        self.num_features = 1 + dynamic.shape[1] + static.shape[1]
        train_end = len(speeds) * 7 // 10
        val_end = len(speeds) * 8 // 10
        splits = (slice(0, train_end), slice(train_end, val_end), slice(val_end, len(speeds)))
        if min(split.stop - split.start for split in splits) < seq_len + pre_len:
            raise ValueError("Every split must contain at least one full forecast window")

        self.feat_max_val = float(np.nanmax(speeds[:train_end]))
        if not np.isfinite(self.feat_max_val) or self.feat_max_val <= 0:
            raise ValueError("The training split needs observed speeds above zero")
        self.dynamic_scale = np.abs(dynamic[:train_end]).max(axis=0)
        self.static_scale = np.abs(static).max(axis=0)
        self.dynamic_scale[self.dynamic_scale == 0] = 1
        self.static_scale[self.static_scale == 0] = 1
        static = static / self.static_scale

        self.train_dataset, self.val_dataset, self.test_dataset = [
            METRLAContextDataset(
                speeds[split] / self.feat_max_val,
                dynamic[split] / self.dynamic_scale,
                static,
                seq_len,
                pre_len,
            )
            for split in splits
        ]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
