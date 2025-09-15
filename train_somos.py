"""
Largely adapt codes from:
https://github.com/nii-yamagishilab/mos-finetune-ssl
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DIR"] = os.path.expanduser("~/wandb")
os.environ["WANDB_CACHE_DIR"] = os.path.expanduser("~/.cache/wandb")

import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.stats
import random
import laion_clap
import heapq
import pandas as pd

from module import MosPredictor
from dataset import SOMOS
from loss import LearnableAdaptiveWeighting, pairwise_ranking_loss, listwise_ranking_loss

# --------------------------
# Fix random seeds
# --------------------------
random.seed(1984)
np.random.seed(1984)
torch.manual_seed(1984)
torch.cuda.manual_seed_all(1984)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# --------------------------
# Main training
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default="/share/nas169/jethrowang/SOMOS", required=False, help='Path of musiceval dataset')
    parser.add_argument('--expname', type=str, required=False, default='exp', help='ckpt will be saved in ../ckpt/EXPNAME')
    parser.add_argument('--pairwise_preference_factor', type=float, required=False, default=7.0)
    parser.add_argument('--pairwise_margin_scale', type=float, required=False, default=0.2)
    parser.add_argument('--listwise_preference_factor', type=float, required=False, default=0.5)
    parser.add_argument('--listwise_temperature', type=float, required=False, default=2.0)
    parser.add_argument('--listwise_margin_scale', type=float, required=False, default=0.3)
    args = parser.parse_args()

    DATA_DIR = args.datadir
    EXP_NAME = args.expname
    CKPT_DIR = './ckpt/somos/' + EXP_NAME

    wandb.init(entity="jethrowang0531", project="PAGFAR-MOS_SOMOS", name=EXP_NAME)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # -----------------------
    # Load upstream model
    # -----------------------
    SAMPLE_RATE = 16000
    UPSTREAM_OUT_DIM = 512
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', tmodel='roberta')
    model.load_ckpt('./upstream/music_speech_audioset_epoch_15_esc_89.98.pt')

    # -----------------------
    # Model & Optimizer
    # -----------------------
    net = MosPredictor('CLAP-music', model, UPSTREAM_OUT_DIM, device, False).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-2)
    weighting_module = LearnableAdaptiveWeighting().to(device)
    pointwise_loss = nn.HuberLoss()

    # -----------------------
    # Dataset & DataLoader
    # -----------------------
    trainset = SOMOS(root_dir=DATA_DIR, split='train', subset='clean', sample_rate=SAMPLE_RATE)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

    testset = SOMOS(root_dir=DATA_DIR, split='valid', subset='clean', sample_rate=SAMPLE_RATE)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, collate_fn=testset.collate_fn)

    top_ckpts = []

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(1, 101):
        # ----- Training -----
        net.train()
        train_epoch_loss = 0.0
        for data in tqdm(trainloader, desc=f"Epoch {epoch} Training", ncols=100):
            wavs, labels, texts, filenames = data
            wavs = wavs.squeeze(1).to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            output = net(wavs, texts)

            weights = weighting_module(labels, output)

            p_loss = pointwise_loss(output, labels)
            pr_loss = pairwise_ranking_loss(output, labels)
            lr_loss = listwise_ranking_loss(output, labels)
            train_loss = weights[0] * p_loss + weights[1] * pr_loss + weights[2] * lr_loss

            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()

            wandb.log({
                "train_loss/total": train_loss.item(),
                "train_loss/pointwise": (weights[0] * p_loss).item(),
                "train_loss/pairwise": (weights[1] * pr_loss).item(),
                "train_loss/listwise": (weights[2] * lr_loss).item(),
                "weights/pointwise": weights[0].item(),
                "weights/pairwise": weights[1].item(),
                "weights/listwise": weights[2].item(),
            })

        avg_train_loss = train_epoch_loss / len(trainloader)
        print(f'EPOCH {epoch} AVG TRAIN LOSS: {avg_train_loss:.4f}')
        wandb.log({'epoch': epoch, 'avg_train_loss': avg_train_loss})

        torch.cuda.empty_cache()

        # --------------------------
        # Test system-level SRCC evaluation
        # --------------------------
        net.eval()
        predictions = {}
        for data in tqdm(testloader, desc=f"Testing Progress", ncols=100):
            wavs, labels, texts, filenames = data
            wavs = wavs.squeeze(1).to(device)
            with torch.no_grad():
                output = net(wavs, texts)
            score = output.cpu().numpy().item()
            utt_id = filenames[0]
            predictions[utt_id] = score

        sys2scores = {}
        for utt_id, score in predictions.items():
            sysID = utt_id.replace(".wav","").split("_")[-1]
            sys2scores.setdefault(sysID, []).append(score)
        pred_sys = {k: np.mean(v) for k,v in sys2scores.items()}

        # ground truth
        truth_file = os.path.join(DATA_DIR, "training_files/split1/clean/test_system.csv")
        df = pd.read_csv(truth_file)
        truth_sys = dict(zip(df["systemId"].astype(str), df["mean"]))

        common_keys = sorted(set(truth_sys.keys()) & set(pred_sys.keys()))
        t = np.array([truth_sys[k] for k in common_keys])
        p = np.array([pred_sys[k] for k in common_keys])
        srcc = scipy.stats.spearmanr(t, p)[0]
        print(f'EPOCH {epoch} SYSTEM SRCC: {srcc:.4f}')
        wandb.log({'system_srcc': srcc})

        ckpt_path = os.path.join(CKPT_DIR, f'ckpt_epoch_{epoch}_srcc_{srcc:.4f}')
        torch.save(net.state_dict(), ckpt_path)
        heapq.heappush(top_ckpts, (srcc, ckpt_path))
        if len(top_ckpts) > 3:
            worst_srcc, worst_path = heapq.heappop(top_ckpts)
            if os.path.exists(worst_path):
                os.remove(worst_path)

    print("\nTop-3 SRCC checkpoints:")
    for s, p in sorted(top_ckpts, reverse=True):
        print(f"SRCC: {s:.4f}, Path: {p}")

    wandb.finish()


if __name__ == '__main__':
    main()
