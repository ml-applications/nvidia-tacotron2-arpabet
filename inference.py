#!/usr/bin/env python

# 1
#import matplotlib
#import matplotlib.pylab as plt

#import IPython.display as ipd

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
checkpoint_path = "/home/bt/models/tacotron2-nvidia/tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

# 5
#waveglow_path = 'waveglow_256channels.pt'
#waveglow = torch.load(waveglow_path)['model']
#waveglow.cuda().eval().half()
#for k in waveglow.convinv:
#    k.float()
#denoiser = Denoiser(waveglow)

# 6
text = "Waveglow is really awesome!"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

# 7
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
#plot_data((mel_outputs.float().data.cpu().numpy()[0],
#           mel_outputs_postnet.float().data.cpu().numpy()[0],
#           alignments.float().data.cpu().numpy()[0].T))

# 8
with torch.no_grad():
    #audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    pass
#ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

