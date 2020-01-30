import torch
import torch.nn as nn
import torch.nn.functional as F

from .res_stack import ResStack
# from res_stack import ResStack

import numpy as np

from .melgan_wav_images import render_histogram
from .melgan_wav_images import rescale_mel

#from wav_images import congrid
import skimage
from skimage.transform import resize

MAX_WAV_VALUE = 32768.0


class Generator(nn.Module):
    def __init__(self, mel_channel):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel

        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channel, 512, kernel_size=7, stride=1)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4)),

            ResStack(256),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)),

            ResStack(128),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)),

            ResStack(64),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),

            ResStack(32),

            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=7, stride=1)),
            nn.Tanh(),
        )

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        # TODO/NB: The actual shapes expected
        # eg, torch.Size([1, 80, 972]), where last dim is dynamic
        # but got: torch.Size([1, 1000, 1025])

        print('>>> Original Tacotron Mel Size:')
        print(mel.shape)

        mel = mel.cpu()

        # Swap
        #mel = np.swapaxes(mel, 1, 2)
        #print('>>> Swapped:')
        print(mel.shape)
        #render_histogram(mel.cpu(), 'histograms/swapped.png', row_index=1, col_index=2)

        # Resize
        new_dims = (1, 80, mel.shape[2])
        #new_dims = (1, mel.shape[1], 80)
        #TODO TODO TODOmel = resize(mel.cpu(), new_dims)
        #mel = np.resize(mel.cpu(), (1, 80, mel.shape[2])) # works, but physical appearance is wrong
        #mel = np.resize(mel.cpu(), (1, mel.shape[1], 80))

        #print('>>> Mel Size (resized):')
        #print(mel.shape)
        #render_histogram(mel, 'histograms/resized.png', row_index=1, col_index=2)

        # RESCALE
        #print('>>> RESCALE MEL:')
        #rescale_mel(mel, -14.0, -0.1)

        # Swap 2
        #mel = np.swapaxes(mel, 1, 2)
        #print('>>> Swapped:')
        #print(mel.shape)
        #render_histogram(mel, 'histograms/swapped2.png', row_index=1, col_index=2)

        hop_length = 256
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        #zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        zero = torch.full((1, self.mel_channel, 10), -11.5129) # TODO Removed device

        #TODO TODO mel = torch.from_numpy(mel)

        #print('>>> Zero Size:')
        #print(zero.shape)

        mel = torch.cat((mel, zero), dim=2)

        #mel = mel.cuda()

        audio = self.forward(mel)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()

        return audio


'''
    to run this, fix
    from . import ResStack
    into
    from res_stack import ResStack
'''
if __name__ == '__main__':
    model = Generator(80)

    x = torch.randn(3, 80, 10)
    print(x.shape)

    y = model(x)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
