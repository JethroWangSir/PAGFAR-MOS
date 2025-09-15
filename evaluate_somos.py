import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import laion_clap
import scipy.stats
from dataset import SOMOS
from module import MosPredictor

def systemID(utt_id: str):
    return utt_id.replace(".wav", "").split("_")[-1]


def read_system_file(filepath):
    df = pd.read_csv(filepath)
    truth = dict(zip(df["systemId"].astype(str), df["mean"]))
    return truth


def aggregate_predictions(predictions):
    sys2scores = {}
    for utt_id, score in predictions.items():
        sysID = systemID(utt_id)
        sys2scores.setdefault(sysID, []).append(score)
    sys_avg = {k: np.mean(v) for k, v in sys2scores.items()}
    return sys_avg


def eval_system_metrics(truth_sys, pred_sys):
    common_keys = sorted(set(truth_sys.keys()) & set(pred_sys.keys()))
    t = np.array([truth_sys[k] for k in common_keys])
    p = np.array([pred_sys[k] for k in common_keys])

    print("\n========== SYSTEM LEVEL ==========")
    mse = np.mean((t - p) ** 2)
    lcc = np.corrcoef(t, p)[0, 1]
    srcc = scipy.stats.spearmanr(t, p)[0]
    ktau = scipy.stats.kendalltau(t, p)[0]
    print(f"MSE  : {mse:.4f}")
    print(f"LCC  : {lcc:.4f}")
    print(f"SRCC : {srcc:.4f}")
    print(f"KTAU : {ktau:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default="/share/nas169/jethrowang/SOMOS")
    parser.add_argument('--ckptdir', type=str, default='./ckpt/somos/exp/ckpt')
    parser.add_argument('--expname', type=str, default='./evaluation/somos/exp')
    parser.add_argument('--truth', type=str, default='/share/nas169/jethrowang/SOMOS/training_files/split1/clean/test_system.csv', help='Ground truth system-level file')
    args = parser.parse_args()

    os.makedirs(args.expname, exist_ok=True)
    outfile = os.path.join(args.expname, 'answer.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAMPLE_RATE = 16000
    UPSTREAM_MODEL = 'CLAP-music'
    UPSTREAM_OUT_DIM = 512
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt('./upstream/music_speech_audioset_epoch_15_esc_89.98.pt')

    net = MosPredictor(UPSTREAM_MODEL, model, UPSTREAM_OUT_DIM, device, False).to(device)
    net.eval()

    ckpt = torch.load(args.ckptdir, map_location=device)
    net.load_state_dict(ckpt)

    # Dataset
    test_set = SOMOS(root_dir=args.datadir, split='test', subset='clean', sample_rate=SAMPLE_RATE)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=test_set.collate_fn)

    predictions = {}
    print('Starting evaluation...')
    for wav, _, text, utt_id in tqdm(test_loader, ncols=100):
        utt_id = utt_id[0]
        wav = wav.squeeze(1).to(device)
        with torch.no_grad():
            output = net(wav, text)
        score = output.cpu().numpy().item()
        predictions[utt_id] = score

    pred_sys = aggregate_predictions(predictions)

    with open(outfile, 'w') as f:
        for sysID in sorted(pred_sys.keys()):
            f.write(f"{sysID},{pred_sys[sysID]}\n")

    truth_sys = read_system_file(args.truth)
    eval_system_metrics(truth_sys, pred_sys)


if __name__ == '__main__':
    main()
