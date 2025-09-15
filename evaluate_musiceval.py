import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import laion_clap
from module import MosPredictor
from utils import get_texts_from_filename
from muq import MuQMuLan
import scipy.stats
from dataset import MusicEvalMetric


def systemID(wavID):
    return wavID.replace("audiomos2025-track1-", "").split('_')[0]


def read_file(filepath):
    """
    Read file with format: wavID,overall,textual
    Returns: dict[wavID] = (overall, textual)
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            wavID, overall, textual = parts
            data[wavID] = (float(overall), float(textual))
    return data


def eval_metrics(truth_dict, pred_dict, label):
    keys = sorted(set(truth_dict.keys()) & set(pred_dict.keys()))
    truth_overall = np.array([truth_dict[k][0] for k in keys])
    truth_textual = np.array([truth_dict[k][1] for k in keys])
    pred_overall = np.array([pred_dict[k][0] for k in keys])
    pred_textual = np.array([pred_dict[k][1] for k in keys])

    print(f"\n========== {label.upper()} ==========")
    for name, t, p in [
        ("OVERALL", truth_overall, pred_overall),
        ("TEXTUAL", truth_textual, pred_textual)
    ]:
        mse = np.mean((t - p) ** 2)
        lcc = np.corrcoef(t, p)[0, 1]
        srcc = scipy.stats.spearmanr(t, p)[0]
        ktau = scipy.stats.kendalltau(t, p)[0]
        print(f"---- {name} ----")
        print(f"MSE  : {mse:.4f}")
        print(f"LCC  : {lcc:.4f}")
        print(f"SRCC : {srcc:.4f}")
        print(f"KTAU : {ktau:.4f}")


def aggregate_by_system(data_dict):
    """
    Returns:
        overall: dict[systemID] = avg_overall
        textual: dict[systemID] = avg_textual
    """
    system_scores_overall = {}
    system_scores_textual = {}
    for wavID, (o, t) in data_dict.items():
        sysID = systemID(wavID)
        system_scores_overall.setdefault(sysID, []).append(o)
        system_scores_textual.setdefault(sysID, []).append(t)

    avg_overall = {k: np.mean(v) for k, v in system_scores_overall.items()}
    avg_textual = {k: np.mean(v) for k, v in system_scores_textual.items()}
    return avg_overall, avg_textual


def eval_system_metrics(truth_dict, pred_dict):
    truth_overall, truth_textual = aggregate_by_system(truth_dict)
    pred_overall, pred_textual = aggregate_by_system(pred_dict)

    common_keys = sorted(set(truth_overall.keys()) & set(pred_overall.keys()))
    truth_o = np.array([truth_overall[k] for k in common_keys])
    pred_o = np.array([pred_overall[k] for k in common_keys])
    truth_t = np.array([truth_textual[k] for k in common_keys])
    pred_t = np.array([pred_textual[k] for k in common_keys])

    print("\n========== SYSTEM LEVEL ==========")
    for name, t, p in [
        ("OVERALL", truth_o, pred_o),
        ("TEXTUAL", truth_t, pred_t)
    ]:
        mse = np.mean((t - p) ** 2)
        lcc = np.corrcoef(t, p)[0, 1]
        srcc = scipy.stats.spearmanr(t, p)[0]
        ktau = scipy.stats.kendalltau(t, p)[0]
        print(f"---- {name} ----")
        print(f"MSE  : {mse:.4f}")
        print(f"LCC  : {lcc:.4f}")
        print(f"SRCC : {srcc:.4f}")
        print(f"KTAU : {ktau:.4f}")


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=False, default="/share/nas169/wago/AudioMOS/data/track1/audiomos2025-track1-eval-phase")
    parser.add_argument('--ckptdir', type=str, required=False, default='./ckpt/musiceval/exp/ckpt')
    parser.add_argument('--expname', type=str, required=False, default='./evaluation/musiceval/exp')
    parser.add_argument('--truth', type=str, required=False, default='/share/nas169/wago/AudioMOS/data/track1/MusicEval-full/sets/test_mos_list.txt', help='Ground truth file path')
    args = parser.parse_args()

    UPSTREAM_MODEL = 'CLAP-music'  # CLAP-music / MuQ-MuLan
    DATADIR = args.datadir
    finetuned_checkpoint = args.ckptdir

    os.makedirs(args.expname, exist_ok=True)
    outfile = os.path.join(args.expname, 'answer.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if UPSTREAM_MODEL == 'CLAP-music':
        SAMPLE_RATE = 16000
        UPSTREAM_OUT_DIM= 512
        model = laion_clap.CLAP_Module(enable_fusion=False,  amodel='HTSAT-base')
        model.load_ckpt('./upstream/music_audioset_epoch_15_esc_90.14.pt')
    elif UPSTREAM_MODEL == 'MuQ-MuLan':
        SAMPLE_RATE = 24000
        UPSTREAM_OUT_DIM= 512
        model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
        model = model.eval()
    else:
        print('*** ERROR *** Model type ' + UPSTREAM_MODEL + ' not supported.')
        exit()

    net = MosPredictor(UPSTREAM_MODEL, model, UPSTREAM_OUT_DIM, device).to(device)
    net.eval()

    ckpt = torch.load(finetuned_checkpoint, map_location=lambda storage, loc: storage.cuda() if device.type == 'cuda' else storage)
    net.load_state_dict(ckpt)

    wavdir = os.path.join(DATADIR, 'DATA/wav')
    test_list = os.path.join(DATADIR, 'DATA/sets/eval_list.txt')

    test_set = MusicEvalMetric(wavdir, test_list, SAMPLE_RATE)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=test_set.collate_fn)

    predictions_overall = {}
    predictions_textual = {}

    print('Starting evaluation')
    for wav, filenames in tqdm(test_loader, ncols=100):
        wav = wav.squeeze(1).to(device)
        text = get_texts_from_filename(filenames)
        with torch.no_grad():
            output1, output2 = net(wav, text)
        output1 = output1.cpu().numpy()[0][0]
        output2 = output2.cpu().numpy()[0][0]
        predictions_overall[filenames[0]] = output1
        predictions_textual[filenames[0]] = output2

    with open(outfile, 'w') as f:
        for fname in sorted(predictions_overall.keys()):
            outl = fname.replace('.wav', '') + ',' + str(predictions_overall[fname]) + ',' + str(predictions_textual[fname]) + '\n'
            f.write(outl)
    
    
    truth = read_file(args.truth)
    pred = read_file(outfile)
    eval_metrics(truth, pred, "Utterance Level")
    eval_system_metrics(truth, pred)

if __name__ == '__main__':
    main()
