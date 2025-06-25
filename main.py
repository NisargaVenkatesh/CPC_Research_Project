import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from cpc.encoder import Encoder
from cpc.ar_model import AutoregressiveModel
from cpc.loss import CPCLoss
from dataset import AudioDataset

# === Setup Logging ===
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/train_{timestamp}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# === Load Config ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# === Initialize Model Components ===
encoder = Encoder().to(device)
ar_model = AutoregressiveModel(input_dim=512, hidden_dim=512).to(device)
cpc_loss = CPCLoss(prediction_steps=config["prediction_steps"]).to(device)

params = list(encoder.parameters()) + list(ar_model.parameters())
optimizer = optim.Adam(params, lr=float(config["learning_rate"]))

# === Load Dataset ===
dataset = AudioDataset(segment_len=config["segment_len"], max_files=1000)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# === Checkpoints Directory ===
os.makedirs("checkpoints", exist_ok=True)

# === Optional Resume Logic ===
start_epoch = 0
resume_epoch = None  # Set to an integer if resuming

if resume_epoch is not None:
    encoder.load_state_dict(torch.load(f"checkpoints/encoder_epoch_{resume_epoch}.pth"))
    ar_model.load_state_dict(torch.load(f"checkpoints/ar_model_epoch_{resume_epoch}.pth"))
    start_epoch = resume_epoch
    logging.info(f"Resumed training from epoch {resume_epoch}")

# === Training Loop ===
epoch_losses = []
best_loss = float("inf")

for epoch in range(start_epoch, config["epochs"]):
    encoder.train()
    ar_model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        waveforms, _ = batch
        waveforms = waveforms.to(device)

        z = encoder(waveforms)
        c = ar_model(z)
        loss = cpc_loss(z, c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # Save current model
    torch.save(encoder.state_dict(), f"checkpoints/encoder_epoch_{epoch + 1}.pth")
    torch.save(ar_model.state_dict(), f"checkpoints/ar_model_epoch_{epoch + 1}.pth")
    logging.info(f"Saved checkpoint for epoch {epoch + 1}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(encoder.state_dict(), "encoder_best.pth")
        torch.save(ar_model.state_dict(), "ar_model_best.pth")
        logging.info(f"Best model updated at epoch {epoch + 1} with loss {avg_loss:.4f}")

# === Save Final Model ===
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(ar_model.state_dict(), "ar_model.pth")
logging.info("Saved final encoder and AR model.")

# === Plot Loss Curve ===
plt.plot(epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("InfoNCE Loss")
plt.title("CPC Training Loss Over Epochs")
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()
logging.info("Saved training_loss.png")
