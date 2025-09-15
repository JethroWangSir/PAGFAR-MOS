"""
Largely adapt codes from:
https://github.com/nii-yamagishilab/mos-finetune-ssl
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from muq import MuQMuLan

from module import MosPredictor
from dataset import MusicEval, MusicEvalMetric
from utils import get_texts_from_filename
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
# Utility functions
# --------------------------
def read_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            wavID, overall, textual = parts
            wavID = wavID.replace('.wav','')  # <- 去掉 .wav
            data[wavID] = (float(overall), float(textual))
    return data

def systemID(wavID):
    return wavID.replace("audiomos2025-track1-", "").split('_')[0]

def aggregate_by_system(pred_overall, pred_textual, truth_dict):
    sys_pred_o, sys_pred_t = {}, {}
    sys_truth_o, sys_truth_t = {}, {}
    for wavID in pred_overall.keys():
        sysID = systemID(wavID)
        sys_pred_o.setdefault(sysID, []).append(pred_overall[wavID])
        sys_pred_t.setdefault(sysID, []).append(pred_textual[wavID])
        sys_truth_o.setdefault(sysID, []).append(truth_dict[wavID][0])
        sys_truth_t.setdefault(sysID, []).append(truth_dict[wavID][1])
    sys_pred_o = {k: np.mean(v) for k, v in sys_pred_o.items()}
    sys_pred_t = {k: np.mean(v) for k, v in sys_pred_t.items()}
    sys_truth_o = {k: np.mean(v) for k, v in sys_truth_o.items()}
    sys_truth_t = {k: np.mean(v) for k, v in sys_truth_t.items()}
    return sys_pred_o, sys_pred_t, sys_truth_o, sys_truth_t

# --------------------------
# Main training
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default="/share/nas169/wago/AudioMOS/data/track1/MusicEval-phase1")
    parser.add_argument('--expname', type=str, default='exp')
    parser.add_argument('--pairwise_preference_factor', type=float, default=7.0)
    parser.add_argument('--pairwise_margin_scale', type=float, default=0.2)
    parser.add_argument('--listwise_preference_factor', type=float, default=0.5)
    parser.add_argument('--listwise_temperature', type=float, default=2.0)
    parser.add_argument('--listwise_margin_scale', type=float, default=0.3)
    parser.add_argument('--test_truth', type=str, default='/share/nas169/wago/AudioMOS/data/track1/MusicEval-full/sets/test_mos_list.txt')
    args = parser.parse_args()

    DATA_DIR = args.datadir
    UPSTREAM_MODEL = 'CLAP-music'  # CLAP-music / MuQ-MuLan
    EXP_NAME = args.expname
    CKPT_DIR = './ckpt/musiceval/' + EXP_NAME  # checkpoint will be saved here

    wandb.init(entity="jethrowang0531", project="PAGFAR-MOS_MusicEval", name=EXP_NAME)

    os.makedirs(CKPT_DIR, exist_ok=True)

    # -----------------------
    # Load upstream model
    # -----------------------
    if UPSTREAM_MODEL == 'CLAP-music':
        SAMPLE_RATE = 16000
        UPSTREAM_OUT_DIM= 512
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', tmodel='roberta')
        model.load_ckpt('./upstream/music_audioset_epoch_15_esc_90.14.pt')
    elif UPSTREAM_MODEL == 'MuQ-MuLan':
        SAMPLE_RATE = 24000
        UPSTREAM_OUT_DIM= 512
        model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
        model = model.eval()
    else:
        print('*** ERROR *** Model type ' + UPSTREAM_MODEL + ' not supported.')
        exit()
    
    # -----------------------
    # Model & Optimizer
    # -----------------------
    net = MosPredictor(UPSTREAM_MODEL, model, UPSTREAM_OUT_DIM, device).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-2)

    weighting_module1 = LearnableAdaptiveWeighting().to(device)
    weighting_module2 = LearnableAdaptiveWeighting().to(device)
    pointwise_loss = nn.HuberLoss()

    # -----------------------
    # Dataset & DataLoader
    # -----------------------
    trainwavdir = os.path.join(DATA_DIR, 'wav')
    trainlist = os.path.join(DATA_DIR, 'sets/train_mos_list.txt')
    textwavdir = '/share/nas169/wago/AudioMOS/data/track1/audiomos2025-track1-eval-phase/DATA/wav'
    testlist = '/share/nas169/wago/AudioMOS/data/track1/audiomos2025-track1-eval-phase/DATA/sets/eval_list.txt'

    trainset = MusicEval(trainwavdir, trainlist, SAMPLE_RATE)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)
    testset = MusicEvalMetric(textwavdir, testlist, SAMPLE_RATE)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, collate_fn=testset.collate_fn)

    truth_dict = read_file(args.test_truth)

    top_ckpts = []

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(1, 101):
        STEPS = 0
        net.train()
        train_epoch_loss = 0.0
        train_epoch_loss1 = 0.0
        train_epoch_loss2 = 0.0
        for i, data in enumerate(tqdm(trainloader, desc="Training Progress", ncols=100), 0):
            STEPS += 1
            wavs, labels1, labels2, filenames = data
            wavs = wavs.squeeze(1).to(device)
            texts = get_texts_from_filename(filenames)

            labels1 = labels1.unsqueeze(1).to(device)
            labels2 = labels2.unsqueeze(1).to(device)

            optimizer.zero_grad()
            output1, output2 = net(wavs, texts)

            weights1 = weighting_module1(labels1, output1)
            weights2 = weighting_module2(labels2, output2)

            pointwise_loss1 = pointwise_loss(output1, labels1)
            pairwise_loss1 = pairwise_ranking_loss(output1, labels1)
            listwise_loss1 = listwise_ranking_loss(output1, labels1)
            loss1 = (
                weights1[0] * pointwise_loss1 +
                weights1[1] * pairwise_loss1 +
                weights1[2] * listwise_loss1
            )

            pointwise_loss2 = pointwise_loss(output2, labels2)
            pairwise_loss2 = pairwise_ranking_loss(output2, labels2)
            listwise_loss2 = listwise_ranking_loss(output2, labels2)
            loss2 = (
                weights2[0] * pointwise_loss2 +
                weights2[1] * pairwise_loss2 +
                weights2[2] * listwise_loss2
            )

            train_loss = (loss1 + loss2) / 2
            loss1.backward(retain_graph=True)
            loss2.backward()

            optimizer.step()

            wandb.log({
                "train_loss1/total": loss1,
                "train_loss1/pointwise": (weights1[0] * pointwise_loss1).item(),
                "train_loss1/pairwise": (weights1[1] * pairwise_loss1).item(),
                "train_loss1/listwise": (weights1[2] * listwise_loss1).item(),

                "train_loss2/total": loss2,
                "train_loss2/pointwise": (weights2[0] * pointwise_loss2).item(),
                "train_loss2/pairwise": (weights2[1] * pairwise_loss2).item(),
                "train_loss2/listwise": (weights2[2] * listwise_loss2).item(),

                "weights1/pointwise": (weights1[0]).item(),
                "weights1/pairwise": (weights1[1]).item(),
                "weights1/listwise": (weights1[2]).item(),

                "weights2/pointwise": (weights2[0]).item(),
                "weights2/pairwise": (weights2[1]).item(),
                "weights2/listwise": (weights2[2]).item(),
            })

            train_epoch_loss += train_loss.item()
            train_epoch_loss1 += loss1.item()
            train_epoch_loss2 += loss2.item()
        print('EPOCH ' + str(epoch) + ', AVG EPOCH TRAIN LOSS: ' + str(train_epoch_loss / STEPS))
        wandb.log({
            'epoch': epoch,
            'avg_train_loss': train_epoch_loss / STEPS,
        })
        
        # clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        # --------------------------
        # Test system-level SRCC evaluation
        # --------------------------
        net.eval()
        predictions_overall, predictions_textual = {}, {}
        for wav, filenames in tqdm(testloader, desc="Testing Progress", ncols=100):
            wav = wav.squeeze(1).to(device)
            texts = get_texts_from_filename(filenames)
            with torch.no_grad():
                out1, out2 = net(wav, texts)
            out1 = out1.cpu().numpy()[0][0]
            out2 = out2.cpu().numpy()[0][0]
            fname = filenames[0].replace('.wav','')
            predictions_overall[fname] = out1
            predictions_textual[fname] = out2

        sys_pred_o, sys_pred_t, sys_truth_o, sys_truth_t = aggregate_by_system(predictions_overall, predictions_textual, truth_dict)
        common_keys = sorted(set(sys_pred_o.keys()) & set(sys_truth_o.keys()))
        truth_o = np.array([sys_truth_o[k] for k in common_keys])
        pred_o = np.array([sys_pred_o[k] for k in common_keys])
        truth_t = np.array([sys_truth_t[k] for k in common_keys])
        pred_t = np.array([sys_pred_t[k] for k in common_keys])

        srcc_o = scipy.stats.spearmanr(truth_o, pred_o)[0]
        srcc_t = scipy.stats.spearmanr(truth_t, pred_t)[0]
        avg_srcc = (srcc_o + srcc_t)/2.0

        print(f"EPOCH {epoch} SYSTEM SRCC (avg/overall/textual): {avg_srcc:.4f} / {srcc_o:.4f} / {srcc_t:.4f}")
        wandb.log({
            'system_srcc_avg': avg_srcc,
            'system_srcc_overall': srcc_o,
            'system_srcc_textual': srcc_t
        })

        ckpt_path = os.path.join(CKPT_DIR, f'ckpt_{epoch}_srcc_{avg_srcc:.4f}_{srcc_o:.4f}_{srcc_t:.4f}')
        torch.save(net.state_dict(), ckpt_path)
        top_ckpts.append((avg_srcc, epoch, ckpt_path))

        top_ckpts = sorted(top_ckpts, key=lambda x: x[0], reverse=True)
        if len(top_ckpts) > 3:
            for srcc_old, epoch_old, path_old in top_ckpts[3:]:
                if os.path.exists(path_old):
                    os.remove(path_old)
            top_ckpts = top_ckpts[:3]

    print("Top 3 checkpoints by avg SRCC:")
    for srcc, epoch, path in top_ckpts:
        print(f"Epoch {epoch}, Avg SRCC: {srcc:.4f}, Path: {path}")

    wandb.finish()


if __name__ == '__main__':
    main()