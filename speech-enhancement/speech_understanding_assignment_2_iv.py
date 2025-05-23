# -*- coding: utf-8 -*-
"""Speech-Understanding-Assignment-2-IV.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EdQWXgw8TIz9j8_ciGdofXw1uJKuxB6S
"""

#!fusermount -u /content/drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)



import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import glob
import itertools
from speechbrain.pretrained import SepformerSeparation as Sepformer
from mir_eval.separation import bss_eval_sources

from transformers import WavLMModel, WavLMConfig
from torchaudio.transforms import Resample
from itertools import permutations

class MultiSpeakerDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000):
        self.sample_rate = sample_rate
        self.examples = []
        folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for folder in folders:
            mix_path = glob.glob(os.path.join(folder, "*-*.wav"))[0]  # mixture file
            clean_paths = sorted([f for f in glob.glob(os.path.join(folder, "*.wav")) if f != mix_path])
            self.examples.append((mix_path, clean_paths))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        mix_path, clean_paths = self.examples[idx]
        mix_waveform, _ = torchaudio.load(mix_path)
        clean_waveforms = [torchaudio.load(p)[0] for p in clean_paths]
        return mix_waveform, clean_waveforms

class SepSIDModel(nn.Module):
    def __init__(self, sepformer, sid_model):
        super().__init__()
        self.sepformer = sepformer
        self.sid_model = sid_model

    def forward(self, mixture):
    # Get estimated sources
      est_sources = self.sepformer.separate_batch(mixture)  # Shape: [batch, time, 2] or [time, 2]

      # Ensure shape is [2, T] for downstream loss
      if est_sources.ndim == 3:
          est_sources = est_sources.squeeze(0)  # from [1, T, 2] to [T, 2]

      # Transpose to [2, T] (permute speakers to first dim)
      est_sources = est_sources.transpose(0, 1).contiguous()

      return est_sources  # Final shape: [2, T]

def compute_speaker_loss(est_embs, ref_embs):
    """
    Computes the speaker loss using cosine similarity between embeddings.

    Args:
        est_embs (list of torch.Tensor): List of estimated speaker embeddings.
        ref_embs (list of torch.Tensor): List of reference speaker embeddings.

    Returns:
        torch.Tensor: The speaker loss.
    """
    cos_sim = nn.CosineSimilarity(dim=-1)
    total_sim = 0
    for i in range(len(ref_embs)):
        sim = 1 - cos_sim(est_embs[i], ref_embs[i])
        # sim.requires_grad = True  # Remove this line - 'sim' is not a leaf variable
        total_sim += sim

    loss = total_sim / len(ref_embs)
    # loss.requires_grad = True  # Remove this line - 'loss' is not a leaf variable

    return loss

def best_perm_loss(est_sources, ref_sources):
    perms = list(itertools.permutations(range(len(ref_sources))))
    min_loss = float("inf")
    best_perm = None
    for perm in perms:
        loss = 0
        for i, j in enumerate(perm):
            est = est_sources[i].squeeze()  # Ensure shape [T]
            ref = ref_sources[j].squeeze()  # Ensure shape [T]
            loss += sisnr_loss(ref, est)
        if loss < min_loss:
            min_loss = loss
            best_perm = perm
    return min_loss, best_perm

def sisnr_loss(source, estimate):
  """
  Computes the SI-SNR (Scale-Invariant Signal-to-Noise Ratio) loss.

  Args:
      source (torch.Tensor): The clean source signal.
      estimate (torch.Tensor): The estimated signal.

  Returns:
      torch.Tensor: The SI-SNR loss.
  """
  assert source.shape == estimate.shape, f"Shape mismatch: {source.shape} vs {estimate.shape}"

  def l2_norm(s):
    return torch.norm(s, dim=-1, keepdim=True)

  source = source - torch.mean(source, dim=-1, keepdim=True)
  estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

  s_target = torch.sum(estimate * source, dim=-1, keepdim=True) * source / (l2_norm(source) ** 2 + 1e-8)
  e_noise = estimate - s_target

  # Ensure the result requires gradient
  loss = -10 * torch.log10((l2_norm(s_target) ** 2) / (l2_norm(e_noise) ** 2 + 1e-8))
  loss.requires_grad = True  # Explicitly set requires_grad to True

  return loss

def compute_speaker_loss(est_embs, ref_embs):
    """
    Computes the speaker loss using cosine similarity between embeddings.

    Args:
        est_embs (list of torch.Tensor): List of estimated speaker embeddings.
        ref_embs (list of torch.Tensor): List of reference speaker embeddings.

    Returns:
        torch.Tensor: The speaker loss.
    """
    cos_sim = nn.CosineSimilarity(dim=-1)
    total_sim = 0
    for i in range(len(ref_embs)):
        sim = 1 - cos_sim(est_embs[i], ref_embs[i])
        # sim.requires_grad = True  # Remove this line - 'sim' is not a leaf variable
        total_sim += sim

    loss = total_sim / len(ref_embs)
    # loss.requires_grad = True  # Remove this line - 'loss' is not a leaf variable

    return loss

def extract_sid_embeddings(model, waveforms):
    with torch.no_grad():
        emb_list = []
        for wav in waveforms:
            # Ensure wav is 2D: [1, T]
            # Correctly handle multiple dimensions
            wav = wav.view(1, -1) #Reshape to 2D: [1, T] regardless of original dimensions
            emb = model(wav.to(model.device)).last_hidden_state.mean(dim=1)
            emb_list.append(emb)
    return emb_list

def evaluate_metrics(est_sources, ref_sources):
    # Convert to (2, T)
    est_np = torch.stack(est_sources).detach().cpu().numpy()

    # Clean up reference sources: (1, 1, T) → (T)
    ref_sources_clean = []
    for r in ref_sources:
        if r.ndim == 3:
            r = r.squeeze(0).squeeze(0)
        elif r.ndim == 2:
            r = r.squeeze(0)
        ref_sources_clean.append(r)

    # Now stack them: shape → (2, T)
    try:
        ref_np = torch.stack(ref_sources_clean).detach().cpu().numpy()
    except Exception as e:
        print(f"Stacking Failed: {e}")
        return 0.0, 0.0, 0.0

    # Final shape check
    if ref_np.shape[0] != 2 or est_np.shape[0] != 2:
        print(f"Skipping evaluation: reference shape = {ref_np.shape}, estimate shape = {est_np.shape}")
        return 0.0, 0.0, 0.0

    try:
        from mir_eval.separation import bss_eval_sources
        sdr, sir, sar, _ = bss_eval_sources(ref_np, est_np, compute_permutation=False)
        return sdr.mean(), sir.mean(), sar.mean()
    except Exception as e:
        print(f"Metric Eval Failed: {e}")
        return 0.0, 0.0, 0.0

def compute_rank1_accuracy(sid_model, est_sources, ref_sources):
    est_embs = extract_sid_embeddings(sid_model, est_sources)
    ref_embs = extract_sid_embeddings(sid_model, ref_sources)

    correct = 0
    for i, est in enumerate(est_embs):
        sims = [F.cosine_similarity(est, ref, dim=-1) for ref in ref_embs]
        pred = torch.argmax(torch.tensor(sims))
        if pred == i:
            correct += 1
    return correct / len(ref_sources)

def train(model, dataset, optimizer, sid_model, device):
    model.train()
    sid_model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for mix, cleans in dataloader:
        # Flatten nested mix structure
        if isinstance(mix, (list, tuple)):
            mix = mix[0]

        # Ensure shape [batch, time]
        if mix.dim() == 3 and mix.shape[1] == 1:
            mix = mix.squeeze(1)  # from [1, 1, T] → [1, T]
        mix = mix.to(device)

        # Validate cleans
        # assert isinstance(cleans[0], list), f"Expected list of tensors, got {type(cleans[0])}"
        # cleans = [c.to(device) for c in cleans[0]]
        cleans = [c.to(device) for c in cleans]

        # Forward pass
        est_sources = model(mix)
        est_sources = [s.squeeze(0) for s in est_sources]

        # Loss calculations
        loss_sep, best_perm = best_perm_loss(est_sources, cleans)
        reordered_cleans = [cleans[i] for i in best_perm]

        est_embs = extract_sid_embeddings(sid_model, est_sources)
        ref_embs = extract_sid_embeddings(sid_model, reordered_cleans)
        loss_sid = compute_speaker_loss(est_embs, ref_embs)

        total_loss = loss_sep + loss_sid
        print("total_loss requires_grad:", total_loss.requires_grad)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Metric evaluation
        sdr, sir, sar = evaluate_metrics(est_sources, reordered_cleans)
        acc = compute_rank1_accuracy(sid_model, est_sources, reordered_cleans)

        print(f"Loss: {total_loss.item():.4f} (Sep: {loss_sep.item():.4f}, SID: {loss_sid.item():.4f}) | SDR: {sdr:.2f}, SIR: {sir:.2f}, SAR: {sar:.2f}, Rank-1 Acc: {acc:.2f}")

        #print(f"Loss: {total_loss.item():.4f} (Sep: {loss_sep.item():.4f}, SID: {loss_sid.item():.4f}) | SDR: {sdr:.2f}, SIR: {sir:.2f}, SAR: {sar:.2f}")

# Example Usage
sepformer = Sepformer.from_hparams(source="speechbrain/sepformer-whamr")
sepformer.device = "cuda"  #  Move SpeechBrain model internals to GPU

sid_model = WavLMModel.from_pretrained("microsoft/wavlm-base")

model = SepSIDModel(sepformer, sid_model).to("cuda")  # wrap and move to GPU

root_dir = '/content/drive/My Drive/vox2_test_aac/mix_utterances/train'
dataset = MultiSpeakerDataset(root_dir)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train(model, dataset, optimizer, sid_model, device="cuda")