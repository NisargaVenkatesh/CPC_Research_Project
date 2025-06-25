import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from cpc.encoder import Encoder
from dataset import AudioDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
checkpoint = torch.load("encoder_best.pth", map_location=device)
encoder.load_state_dict(checkpoint)
encoder.eval()

dataset = AudioDataset(segment_len=16000, max_files=100)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

embeddings = []
labels = []
for waveform, label in tqdm(dataloader, desc="Extracting"):
    waveform = waveform.to(device)
    with torch.no_grad():
        z = encoder(waveform)
        embedding = z.mean(dim=1).squeeze(0)
        embeddings.append(embedding.cpu().numpy())
        labels.append(label)

np.save("embeddings/embeddings.npy", np.stack(embeddings))
np.save("embeddings/labels.npy", np.array(labels))