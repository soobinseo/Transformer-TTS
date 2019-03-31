import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse

def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def synthesis(text, args):
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = t.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = t.zeros([1,1, 80]).cuda()
    pos_text = t.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()

    m=m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)
    
    pbar = tqdm(range(args.max_len))
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            mel_input = t.cat([mel_input, postnet_pred[:,-1:,:]], dim=1)

        mag_pred = m_post.forward(postnet_pred)
        
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/test.wav", hp.sr, wav)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=172000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=100000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=400)

    args = parser.parse_args()
    synthesis("Transformer model is so fast!",args)
