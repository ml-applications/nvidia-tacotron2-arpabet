#!/usr/bin/env python

# 1
#import matplotlib
#import matplotlib.pylab as plt
import torchaudio

import IPython.display as ipd

WAVEGLOW_MODEL_PATH = '/home/bt/models/waveglow-nvidia/waveglow_256channels_ljs_v2.pt'

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
#from denoiser import Denoiser

from wav_images import render_histogram

# 2
def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')

# 3
hparams = create_hparams()
hparams.sampling_rate = 22050

# 4
#checkpoint_path = "/home/bt/models/tacotron2-nvidia/tacotron2_statedict.pt"
#checkpoint_path = "/home/bt/models/tacotron2-nvidia/nvidia_pretrained.pt"
checkpoint_path = "/home/bt/models/tacotron2-nvidia/tacotron2_arpabet_ljs_checkpoint_130000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

# 5
#waveglow_path = 'waveglow_256channels.pt'
#waveglow = torch.load(waveglow_path)['model']
waveglow = torch.load(WAVEGLOW_MODEL_PATH)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
#denoiser = Denoiser(waveglow)

# 6
text = "This is the song that never ends. It goes on and on my friend."
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

# 7
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
#plot_data((mel_outputs.float().data.cpu().numpy()[0],
#           mel_outputs_postnet.float().data.cpu().numpy()[0],
#           alignments.float().data.cpu().numpy()[0].T))

# Per melgan preprocess.py, need to output:
# mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
print("Mel outputs:")
print(mel_outputs)
print(mel_outputs.shape)
print(mel_outputs_postnet)
print(mel_outputs_postnet.shape)

print('Saving mels')
torch.save(mel_outputs, 'new_mel_outputs.mel')
torch.save(mel_outputs_postnet, 'new_mel_outputs_postnet.mel')

print('Rendering histograms')
render_histogram(mel_outputs, 'new_mel_outputs.png')
render_histogram(mel_outputs_postnet, 'new_mel_outputs_postnet.png')

# 8
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    pass
#wav = ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
#print(wav)

torchaudio.save('amazing_sound.wav', audio.float().cpu(), hparams.sampling_rate)

