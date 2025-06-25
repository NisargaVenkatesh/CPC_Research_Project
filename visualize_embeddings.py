import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from cpc.encoder import Encoder
from dataset import AudioDataset

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained encoder
encoder = Encoder()
encoder.load_state_dict(torch.load("encoder_best.pth", map_location=device))
encoder.eval().to(device)

# Load dataset
dataset = AudioDataset(segment_len=16000, max_files=500)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Extract embeddings
embeddings = []
labels = []

with torch.no_grad():
    for waveform, speaker_id in dataloader:
        waveform = waveform.to(device)
        z = encoder(waveform)  # [1, T, D]
        z_mean = z.mean(dim=1).squeeze().cpu().numpy()  # [D]
        embeddings.append(z_mean)
        labels.append(speaker_id[0])  # Use string label directly

# Convert to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Print info
print(f"Total embeddings collected: {len(embeddings)}")

# Encode labels for coloring
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Dynamically determine t-SNE perplexity
effective_perplexity = min(30, max(2, len(embeddings) // 3))
print(f"Using t-SNE perplexity: {effective_perplexity}")

# t-SNE
tsne = TSNE(n_components=2, perplexity=effective_perplexity, init='pca', random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=encoded_labels, cmap='tab10', alpha=0.7)
plt.title("t-SNE of CPC Encoded Audio Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(scatter, label="Speaker ID")
plt.tight_layout()
plt.savefig("logs/tsne_embeddings.png")
plt.show()
