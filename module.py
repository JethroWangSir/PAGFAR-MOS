import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModel
from torch import Tensor
from typing import Optional


class MosPredictor(nn.Module):
    def __init__(self, up_model_name, up_model, up_out_dim, device, text_branch=True):
        super(MosPredictor, self).__init__()
        self.upstream_model_name = up_model_name
        self.upstream_model = up_model
        self.upstream_feat_dim = up_out_dim
        self.device = device
        self.text_branch = text_branch

        self.self_attention_wav = SelfAttention(dim=up_out_dim)
        self.cross_attention_wav = CrossAttention(dim=up_out_dim)
        self.gated_fusion_wav = GatedFusion(dim=up_out_dim)

        self.overall_pooling = AttentiveStatisticsPooling(input_dim=up_out_dim)

        self.overall_mlp_layer = nn.Sequential(
            nn.Linear(in_features=self.upstream_feat_dim * 2, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        if self.text_branch:
            self.self_attention_text = SelfAttention(dim=up_out_dim)
            self.cross_attention_text = CrossAttention(dim=up_out_dim)
            self.gated_fusion_text = GatedFusion(dim=up_out_dim)

            self.textual_pooling = AttentiveStatisticsPooling(input_dim=up_out_dim)

            self.textual_mlp_layer = nn.Sequential(
                nn.Linear(in_features=self.upstream_feat_dim * 2, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=1)
            )

    def forward(self, wavs, texts):
        if self.upstream_model_name == 'CLAP-music':
            # feature extraction
            wav_embed_seq = self.upstream_model.get_audio_embedding_from_data(wavs, use_tensor=True).to(self.device)  # [B, T_wav, D]
            text_embed_seq = self.upstream_model.get_text_embedding(texts, use_tensor=True).to(self.device)  # [B, T_text, D]

            # feature fusion
            self_attended_wav = self.self_attention_wav(wav_embed_seq)  # [B, T_wav, D]
            cross_attended_wav = self.cross_attention_wav(wav_embed_seq, text_embed_seq, text_embed_seq)  # [B, T_wav, D]
            fused_wav = self.gated_fusion_wav(self_attended_wav, cross_attended_wav)

            # pooling
            wav_embed = self.overall_pooling(fused_wav)

            if self.text_branch:
                # feature fusion
                self_attended_text = self.self_attention_text(text_embed_seq)  # [B, T_text, D]
                cross_attended_text = self.cross_attention_text(text_embed_seq, wav_embed_seq, wav_embed_seq)  # [B, T_text, D]
                fused_text = self.gated_fusion_text(self_attended_text, cross_attended_text)

                # pooling
                text_embed = self.textual_pooling(fused_text)

        elif self.upstream_model_name == 'MuQ-MuLan':
            wav_embed_seq = self.upstream_model(wavs=wavs).to(self.device)  # [B, T_wav, D]
            text_embed_seq = self.upstream_model(texts=texts).to(self.device)  # [B, T_text, D]
            self_attended_wav = self.self_attention_wav(wav_embed_seq)  # [B, T_wav, D]
            cross_attended_wav = self.cross_attention_wav(wav_embed_seq, text_embed_seq, text_embed_seq)  # [B, T_wav, D]
            # fused_wav = torch.sigmoid(self.alpha) * self_attended_wav + (1 - torch.sigmoid(self.alpha)) * cross_attended_wav
            fused_wav = self.gated_fusion_wav(self_attended_wav, cross_attended_wav)
            wav_embed = self.overall_pooling(fused_wav)

            if self.text_branch:
                self_attended_text = self.self_attention_text(text_embed_seq)  # [B, T_text, D]
                cross_attended_text = self.cross_attention_text(text_embed_seq, wav_embed_seq, wav_embed_seq)  # [B, T_text, D]
                # fused_text = torch.sigmoid(self.beta) * self_attended_text + (1 - torch.sigmoid(self.beta)) * cross_attended_text
                fused_text = self.gated_fusion_text(self_attended_text, cross_attended_text)
                text_embed = self.textual_pooling(fused_text)

        else:
            print('*** ERROR *** Model type ' + self.upstream_model_name + ' not supported.')
            exit()
        
        out1 = self.overall_mlp_layer(wav_embed)
        if self.text_branch:
            out2 = self.textual_mlp_layer(text_embed)
            return out1, out2
        else:
            return out1


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, D], where B is batch size, T is sequence length, D is feature dimension
        attn_output, _ = self.attn(query=x, key=x, value=x)

        return self.norm(attn_output + x)  # residual + norm


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, value):
        # query: [B, T_q, D], key: [B, T_k, D], value: [B, T_k, D]
        attn_output, _ = self.attn(query=query, key=key, value=value)

        return self.norm(attn_output + query)  # residual + norm


class GatedFusion(nn.Module):
    """
    Gated fusion of self-attention and cross-attention outputs.
    """
    def __init__(self, dim, scalar=True):
        super(GatedFusion, self).__init__()
        # Gate weights, one for each channel
        if scalar:
            self.gate = nn.Linear(dim * 2, 1)
        else:
            self.gate = nn.Linear(dim * 2, dim)
    
    def forward(self, self_attn_out, cross_attn_out):
        """
        self_attn_out: [B, T, D]
        cross_attn_out: [B, T, D]
        """
        # Concatenate along feature dimension
        combined = torch.cat([self_attn_out, cross_attn_out], dim=-1)  # [B, T, 2*D]
        # Compute gate value
        gate = torch.sigmoid(self.gate(combined))  # [B, T, D], in (0,1)
        # print(f'Gate: {gate.shape}')
        # Fuse features
        fused = gate * self_attn_out + (1 - gate) * cross_attn_out

        return fused


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentiveStatisticsPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, T, D]
        attn_scores = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T, 1]

        # mean: weighted sum
        mean = torch.sum(attn_weights * x, dim=1)  # [B, D]

        # std: weighted std
        mean_expand = mean.unsqueeze(1)  # [B, 1, D]
        std = torch.sqrt(torch.sum(attn_weights * (x - mean_expand) ** 2, dim=1) + 1e-9)  # [B, D]

        pooled = torch.cat((mean, std), dim=1)  # [B, 2D]

        return pooled
