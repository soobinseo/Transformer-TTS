# Transformer-TTS
A Pytorch Implementation of [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)

<img src="png/model.png">

## Requirements
  * Install python 3
  * Install pytorch == 0.4.0
  * Install requirements:
    ```
   	pip install -r requirements.txt
   	```

## Data
I used LJSpeech dataset which consists of pairs of text script and wav files. The complete dataset (13,100 pairs) can be downloaded [here](https://keithito.com/LJ-Speech-Dataset/). I referred https://github.com/keithito/tacotron and https://github.com/Kyubyong/dc_tts for the preprocessing code.

## Attention images

## Learning curves

## Experimental notes

## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `prepare_data.py` preprocess wav files to mel, linear spectrogram and save them for faster training time. Preprocessing codes for text is in text/ directory.
  * `preprocess.py` includes all preprocessing codes when you loads data.
  * `module.py` contains all methods, including attention, prenet, postnet and so on.
  * `network.py` contains networks including encoder, decoder and post-processing network.
  * `train_transformer.py` is for training autoregressive attention network. (text --> mel)
  * `train_postnet.py` is for training post network. (mel --> linear)
  * `synthesis.py` is for generating TTS sample.

## Training the network
  * STEP 1. Download and extract LJSpeech data at any directory you want.
  * STEP 2. Adjust hyperparameters in `hyperparams.py`, especially 'data_path' which is a directory that you extract files, and the others if necessary.
  * STEP 3. Run `prepare_data.py`.
  * STEP 4. Run `train_transformer.py`.
  * STEP 5. Run `train_postnet.py`.

## Generate TTS wav file
  * STEP 1. Run `synthesis.py`. Make sure the restore step. 

## Samples
  * You can check the generated samples in 'samples/' directory.

## Reference
  * Keith ito: https://github.com/keithito/tacotron
  * Kyubyong Park: https://github.com/Kyubyong/dc_tts
  * jadore801120: https://github.com/jadore801120/attention-is-all-you-need-pytorch/

## Comments
  * Any comments for the codes are always welcome.

