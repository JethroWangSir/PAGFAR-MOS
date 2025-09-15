import os
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import torch.nn.functional as F


class MusicEval(Dataset):
    def __init__(self, wavdir, mos_list, target_sr=16000):
        self.mos_overall_lookup = {}
        self.mos_coherence_lookup = {}
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos_overall = float(parts[1])
            mos_coherence = float(parts[2])
            self.mos_overall_lookup[wavname] = mos_overall
            self.mos_coherence_lookup[wavname] = mos_coherence

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_overall_lookup.keys())
        self.target_sr = target_sr


    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav, sr = torchaudio.load(wavpath)

        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            wav = resampler(wav)

        # Cut to max length (30s * 16000 = 480000 samples / 30s * 24000 = 720000 samples)
        max_samples = self.target_sr * 30
        if wav.size(1) > max_samples:
            wav = wav[:, :max_samples]

        overall_score = self.mos_overall_lookup[wavname]
        coherence_score = self.mos_coherence_lookup[wavname]
        return wav, overall_score, coherence_score, wavname
    
    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):
        wavs, overall_scores, coherence_scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        overall_scores  = torch.stack([torch.tensor(x) for x in list(overall_scores)], dim=0)
        coherence_scores  = torch.stack([torch.tensor(x) for x in list(coherence_scores)], dim=0)
        
        return output_wavs, overall_scores, coherence_scores, wavnames


class MusicEvalMetric(Dataset):
    def __init__(self, wavdir, list_file, target_sr=16000):
        self.wavdir = wavdir
        with open(list_file, 'r') as f:
            self.wavnames = sorted([line.strip().split(',')[0] for line in f])
        self.target_sr = target_sr

    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname + '.wav')
        wav, sr = torchaudio.load(wavpath)

        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            wav = resampler(wav)

        # Cut to max length (30s * 16000 = 480000 samples / 30s * 24000 = 720000 samples)
        max_samples = self.target_sr * 30
        if wav.size(1) > max_samples:
            wav = wav[:, :max_samples]

        return wav, wavname

    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):
        wavs, wavnames = zip(*batch)
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            padded = torch.nn.functional.pad(wav, (0, max_len - wav.shape[1]), 'constant', 0)
            output_wavs.append(padded)
        return torch.stack(output_wavs), wavnames


class SOMOS(Dataset):
    def __init__(self, root_dir, split='train', subset='clean', sample_rate=16000, max_sec=30):
        assert split in ['train', 'valid', 'test']
        assert subset in ['clean', 'full']
        
        self.root_dir = root_dir
        self.split = split
        self.subset = subset
        self.sample_rate = sample_rate
        self.max_samples = int(max_sec * sample_rate)

        training_path = os.path.join(root_dir, 'training_files', 'split1', subset)
        self.mos_file = os.path.join(training_path, f'{split}_mos_list.txt')
        self.audio_dir = os.path.join(root_dir, 'audios')

        self.mos_df = pd.read_csv(self.mos_file, sep=',')
        
        transcript_file = os.path.join(root_dir, 'transcript', 'all_transcripts.txt')
        self.utt2text = {}
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f.read().splitlines():
                if '\t' in line:
                    utt_id, text = line.split('\t', 1)
                    self.utt2text[utt_id] = text
        
        self.mos_df['utt_id_clean'] = self.mos_df['utteranceId'].str.replace('.wav', '', regex=False)
        self.mos_df['text'] = self.mos_df['utt_id_clean'].map(self.utt2text)
        self.mos_df.to_csv(f"/share/nas169/jethrowang/SOMOS/df/{self.subset}_{self.split}.csv", index=False)

    def __len__(self):
        return len(self.mos_df)

    def __getitem__(self, idx):
        row = self.mos_df.iloc[idx]
        utt_id = row['utteranceId']
        utt_id_clean = row['utt_id_clean']
        mos_score = float(row['mean'])
        text = [row['text']]

        wav_file = os.path.join(self.audio_dir, utt_id)
        if not os.path.isfile(wav_file):
            raise FileNotFoundError(f'{wav_file} not found!')

        waveform, sr = torchaudio.load(wav_file)
        if sr != self.sample_rate:
            waveform = T.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        if waveform.size(1) > self.max_samples:
            waveform = waveform[:, :self.max_samples]

        return waveform, mos_score, text, utt_id

    def collate_fn(self, batch):
        waveforms, mos_scores, texts, utt_ids = zip(*batch)

        max_len = max(waveforms, key=lambda x: x.shape[1]).shape[1]
        padded_wavs = []
        for wav in waveforms:
            pad_len = max_len - wav.shape[1]
            padded_wav = F.pad(wav, (0, pad_len))
            padded_wavs.append(padded_wav)
        padded_wavs = torch.stack(padded_wavs, dim=0)

        mos_scores = torch.tensor(mos_scores, dtype=torch.float)

        return padded_wavs, mos_scores, [t[0] for t in texts], list(utt_ids)
