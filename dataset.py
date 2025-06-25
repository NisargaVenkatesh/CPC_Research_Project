import torchaudio
from torch.utils.data import Dataset
import torch
import random

class AudioDataset(Dataset):
    def __init__(self, root="./data", subset="train-clean-100", segment_len=16000, max_files=100):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root, url=subset, download=True)
        self.segment_len = segment_len
        self.indices = list(range(len(self.dataset)))[:max_files]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        waveform, sample_rate, speaker_id, *_ = self.dataset[idx]
        if waveform.size(1) < self.segment_len:
            pad = self.segment_len - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.segment_len]
        return waveform.squeeze(0), speaker_id  # return waveform and label

