# Contrastive Predictive Coding (CPC) for Audio Representation Learning

This project implements **Contrastive Predictive Coding (CPC)** for self-supervised learning of audio representations using PyTorch.

CPC learns meaningful features from raw audio **without labels** by predicting future representations using contrastive loss.

---

## 📦 Project Structure

Res_CPC/
├── cpc/ # Encoder, autoregressive model, CPC loss
├── data/ # LibriSpeech dataset (auto-downloaded)
├── checkpoints/ # Saved model states
├── embeddings/ # Extracted features and labels (npy format)
├── logs/ # Training logs
├── config.yaml # Hyperparameters
├── main.py # Training script
├── extract_embeddings.py # Save encoder features to .npy
├── train_classifier.py # Classify speaker using embeddings
├── visualize_embeddings.py# t-SNE plot of learned embeddings
└── requirements.txt # Python dependencies

---

##  Getting Started

### 1. Create Environment & Install Dependencies

```bash
conda create -n cpc-env python=3.9
conda activate cpc-env
pip install -r requirements.txt

2. Train the CPC Model:
    
    python main.py

Trains encoder and autoregressive model using CPC loss
Logs loss and saves models to checkpoints/
Training hyperparameters are in config.yaml

3. Extract Embeddings:
    After training:

    python extract_embeddings.py

Saves averaged encoder features to embeddings/embeddings.npy
Saves speaker labels to embeddings/labels.npy

4. Visualize Embeddings:

    python visualize_embeddings.py

    Runs t-SNE to reduce 512D embeddings to 2D
    Outputs tsne_embeddings.png to inspect learned structure

5. Evaluate with Classification

    python train_classifier.py

    Trains a simple classifier (Logistic Regression)
    Evaluates performance (e.g., speaker identification)

6. Training Output
    Training loss plot: training_loss.png
    Best model: encoder_best.pth, ar_model_best.pth
    Final model: encoder.pth, ar_model.pth

References
van den Oord et al., 2018. Representation Learning with Contrastive Predictive Coding
LibriSpeech Dataset

Author
Nisarga
