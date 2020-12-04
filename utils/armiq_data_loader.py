import torch
import torch.nn.functional as F
from utils.utils import one_hot_vetorisation



class ARMIQ_TRAIN_DATALOADER_ONEHOT(torch.utils.data.Dataset):
  def __init__(self,samples,masks):
      self.samples = samples
      self.masks = masks


  def __len__(self):
      return len(self.samples)

  def __getitem__(self, idx):
      output = {'event':self.samples[idx], 'mask':self.masks[idx]}
      return output


class ARMIQ_TRAIN_DATALOADER(torch.utils.data.Dataset):
  def __init__(self,samples,masks):
      self.samples = samples
      self.masks = masks



  def __len__(self):
      return len(self.samples)

  def __getitem__(self, idx):
      output = {'event':self.samples[idx], 'mask':self.masks[idx]}
      return output

class ARMIQ_EVAL_DATALOADER(torch.utils.data.Dataset):
  def __init__(self,samples,annotations,masks):
      self.samples = samples
      self.masks = masks
      self.annots = annotations

  def __len__(self):
      return len(self.samples)

  def __getitem__(self, idx):
      output = {'event':self.samples[idx], 'mask':self.masks[idx], 'annot':self.annots[idx]}
      return output