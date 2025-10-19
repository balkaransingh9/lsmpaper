import torch
from torch.utils.data import Dataset
from normaliser import normaliser
import lmdb
import pickle
import pandas as pd

# --Dataset

class MortalityData(Dataset):
  def __init__(self, list_file, split='train', split_col_name='split', lmdb_path='none', norm=None):
    self.list_file = list_file
    self.split = split
    self.split_col_name = split_col_name
    self.lmdb_path = lmdb_path
    self.norm = norm

    if self.split == 'train':
      self.data_split = list_file[list_file[self.split_col_name] == 'train'].reset_index(drop=True)
    elif self.split == 'val':
      self.data_split = list_file[list_file[self.split_col_name] == 'val'].reset_index(drop=True)
    else:
      self.data_split = list_file[list_file[self.split_col_name] == 'test'].reset_index(drop=True)

    self.sample_labels = torch.tensor(self.data_split['In-hospital_death'].values).float()
    self.keys = [s.encode('utf-8') for s in self.data_split['RecordID'].astype(str)]

  def __len__(self):
    return len(self.data_split)

  def __getitem__(self, idx):
      key = self.keys[idx]
      with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
          with env.begin() as txn:
              value_bytes = txn.get(key)
              obj = pickle.loads(value_bytes)

      X = torch.tensor(obj['ffill']).float()
      X = torch.nan_to_num(X, nan=0.0)

      if self.norm is not None:
          mean = self.norm['mean']
          std = self.norm['std']
          std = torch.where(std == 0, torch.ones_like(std), std)  # avoid div by 0
          X = (X - mean) / std

      col_names = obj['columns']
      return X, col_names, self.sample_labels[idx]


# -- Collate --

class MortalityCollate:
  def __init__(self):
    pass
  def __call__(self, batch):
    X_list, col_list, labels_list = zip(*batch)
    X_list = [torch.nan_to_num(X, nan=0.0) for X in X_list]
    X = torch.nn.utils.rnn.pad_sequence(X_list, batch_first=True)
    labels = torch.stack(labels_list)
    return {
        'X': X,
        'labels': labels
    }

# -- Data Module --

import pytorch_lightning as pl
from torch.utils.data import DataLoader

class MortalityDataModule(pl.LightningDataModule):
  def __init__(self, listfile, lmdb_path = 'none',
               batch_size=64, num_workers=4, pin_memory=True,
               prefetch_factor=2, persistent_workers=True, multiprocessing_context=None):
    super().__init__()
    self.listfile = listfile
    self.lmdb_path = lmdb_path

    self.batch_size = batch_size
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.prefetch_factor = prefetch_factor
    self.persistent_workers = persistent_workers
    self.multiprocessing_context = multiprocessing_context

    self.mortality_norm = normaliser(self.listfile, lmdb_path = self.lmdb_path,
                                     split_col = 'split', key_col = 'RecordID',
                                     pickled_value_field = 'ffill')

  def prepare_data(self):
     pass

  def setup(self, stage=None):
    if stage in (None, 'fit'):
      self.train_ds = MortalityData(self.listfile, split='train', lmdb_path=self.lmdb_path, norm = self.mortality_norm)
      self.val_ds = MortalityData(self.listfile, split='val', lmdb_path=self.lmdb_path, norm = self.mortality_norm)
    if stage in (None, 'test'):
      self.test_ds = MortalityData(self.listfile, split='test', lmdb_path=self.lmdb_path, norm = self.mortality_norm)

  def train_dataloader(self):
    return DataLoader(
        self.train_ds,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        multiprocessing_context=self.multiprocessing_context,
        persistent_workers=self.persistent_workers,
        pin_memory=self.pin_memory,
        prefetch_factor=self.prefetch_factor,
        collate_fn=MortalityCollate())

  def val_dataloader(self):
    return DataLoader(
        self.val_ds,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers,
        multiprocessing_context=self.multiprocessing_context,
        persistent_workers=self.persistent_workers,
        pin_memory=self.pin_memory,
        prefetch_factor=self.prefetch_factor,
        collate_fn=MortalityCollate())

  def test_dataloader(self):
    return DataLoader(
        self.test_ds,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers,
        multiprocessing_context=self.multiprocessing_context,
        persistent_workers=self.persistent_workers,
        pin_memory=self.pin_memory,
        prefetch_factor=self.prefetch_factor,
        collate_fn=MortalityCollate())