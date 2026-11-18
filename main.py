# ===========================
# SILENCE WARNINGS
# ===========================
import warnings
warnings.filterwarnings("ignore")

# ===========================
# IMPORTS
# ===========================
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import librosa
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

# ===========================
# CONFIG
# ===========================
DATASET_PATH = "/kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 112
MAX_FRAMES = 16
MAX_AUDIO_LEN = 300
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4

# ===========================
# DATA LOADER
# ===========================
def load_data(limit=1000):
    with open(f"{DATASET_PATH}/metadata.min.json") as f:
        metadata = json.load(f)

    data = []
    for item in metadata[:limit]:
        path = f"{DATASET_PATH}/{item['file']}"
        label = 1 if item['modify_audio'] or item['modify_video'] else 0
        data.append((path, label))

    return data

# ===========================
# VIDEO PROCESSING
# ===========================
def process_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, MAX_FRAMES).astype(int)

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3))
        else:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    frames = np.array(frames)
    return torch.tensor(frames).permute(0, 3, 1, 2).float()

# ===========================
# AUDIO PROCESSING (FIXED)
# ===========================
def process_audio(path):
    y, sr = librosa.load(path, sr=16000)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel = librosa.power_to_db(mel)

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    # PAD/TRUNCATE
    if mel.shape[1] < MAX_AUDIO_LEN:
        pad = MAX_AUDIO_LEN - mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad)))
    else:
        mel = mel[:, :MAX_AUDIO_LEN]

    return torch.tensor(mel).unsqueeze(0).float()

# ===========================
# DATASET
# ===========================
class DeepfakeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        video = process_video(path)
        audio = process_audio(path)

        return video, audio, torch.tensor(label).float()

# ===========================
# MODEL (CROSS MODAL)
# ===========================
class CrossModalModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Video encoder
        self.video_model = models.resnet18(pretrained=True)
        self.video_model.fc = nn.Identity()

        # Audio encoder
        self.audio_model = models.resnet18(pretrained=True)
        self.audio_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.audio_model.fc = nn.Identity()

        # Interaction
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, video, audio):
        B, T, C, H, W = video.shape

        # Process video frames
        video = video.view(B * T, C, H, W)
        v_feat = self.video_model(video)
        v_feat = v_feat.view(B, T, -1).mean(1)

        # Process audio
        a_feat = self.audio_model(audio)

        # CROSS MODAL INTERACTION
        combined = torch.cat([v_feat, a_feat], dim=1)

        return self.fc(combined).squeeze()

# ===========================
# TRAINING
# ===========================
def train():
    data = load_data(limit=1200)

    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(DeepfakeDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DeepfakeDataset(val_data), batch_size=BATCH_SIZE)

    model = CrossModalModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # TRAIN
        model.train()
        losses = []

        for video, audio, label in tqdm(train_loader, leave=False):
            video, audio, label = video.to(DEVICE), audio.to(DEVICE), label.to(DEVICE)

            out = model(video, audio)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Loss: {np.mean(losses):.4f}")

        # VALIDATION
        model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for video, audio, label in val_loader:
                video, audio = video.to(DEVICE), audio.to(DEVICE)

                out = torch.sigmoid(model(video, audio))
                preds.extend((out > 0.5).cpu().numpy())
                gts.extend(label.numpy())

        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds)

        print(f"Val Acc: {acc:.4f} | F1: {f1:.4f}")

    print("\n✅ TRAINING COMPLETE")

# ===========================
# RUN
# ===========================
train()
