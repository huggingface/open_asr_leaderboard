"""
NeMo ASR Model Profiler

This script performs a forward pass on an NeMo ASR models and measures its real-time factor (RTF).
RTF is a metric used to evaluate the processing speed of ASR models.

# audio has to be a mono wav file with 16kHz sample rate

Parameters:
    --model: ASR model name or path to the model checkpoint file.
    --decoding_type: Type of decoding to use (ctc or rnnt).
    --gpu: GPU device to use.
    --batch_size: Batch size to use for inference.
    --nbatches: Total number of batches to process.
    --warmup_batches: Number of batches to skip as warmup.
    --audio: Path to the input audio file for ASR.
    --audio_maxlen: Maximum duration of audio to process (in seconds).

Example:
    python calc_rtf.py --model stt_en_conformer_ctc_large --decoding_type ctc --gpu 0 --batch_size 1 --nbatches 3 --warmup_batches 3 --audio ../data/sample_ami-es2015b.wav --audio_maxlen 30
"""

import time
import argparse
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
import copy
import sys
import soundfile as sf
import numpy as np
import librosa

from nemo.utils import logging
from nemo.collections.asr.models import ASRModel



parser = argparse.ArgumentParser(description='model forward pass profiler / performance tester.')
parser.add_argument("--model", default='stt_en_fastconformer_ctc_large', type=str, help="ASR model")
parser.add_argument("--decoding_type", default='ctc', type=str, help="Type of model [rnnt, ctc, aed]")
parser.add_argument("--gpu", default=0, type=int, help="GPU device to use")
parser.add_argument("--batch_size", default=1, type=int, help="batch size to use")
parser.add_argument("--nbatches", default=3, type=int, help="Total Number of batches to process")
parser.add_argument("--warmup_batches", default=3, type=int, help="Number of batches to skip as warmup")
parser.add_argument("--audio", default="../data/sample_ami-es2015b.wav", type=str, help="wav file to use")

# parser.add_argument("--audio_maxlen", default=30, type=float, help="Multiple chunks of audio of this length is used to calculate RTFX")

args = parser.parse_args()
torch.backends.cudnn.benchmark=True

WAV = args.audio
SAMPLING_RATE = 16000
chunk_len = 30
total_audio_len = 600
MODEL = args.model
batch_size = args.batch_size
nbatches = args.nbatches
warmup_batches = args.warmup_batches
decoding_type = args.decoding_type
total_chunks = int(total_audio_len / chunk_len)

DEVICE=torch.device(args.gpu)

logging.info(f'MODEL: {MODEL}')

def get_samples(audio_file, total_audio_len, target_sr=16000):
    with sf.SoundFile(audio_file, 'r') as f:
        dtype = 'int16'
        sample_rate = f.samplerate
        samples = f.read(dtype=dtype)
        if sample_rate != target_sr:
            samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
        samples = samples.astype('float32') / 32768
        samples = samples.transpose()
        sample_length = samples.shape[0]
        if sample_length > total_audio_len * target_sr:
            logging.info(f'resizing audio sample from {sample_length / target_sr} to maxlen of {total_audio_len}')
            sample_length = int(total_audio_len * target_sr)
            samples = samples[:sample_length]
            logging.info(f'new sample lengh: {samples.shape[0]}')
        else:
            pad_length = int(total_audio_len * target_sr) - sample_length
            logging.info(f'padding audio sample from {sample_length / target_sr} to maxlen of {total_audio_len}')
            samples = np.pad(samples, (0, pad_length), 'constant', constant_values=(0, 0))
            sample_length = int(total_audio_len * target_sr)
            
        return samples, sample_length


def main():

    if MODEL.endswith('.nemo'):
        asr_model = ASRModel.restore_from(MODEL)
    else:
        asr_model = ASRModel.from_pretrained(MODEL)

    asr_model.to(DEVICE)
    asr_model.eval()
    asr_model._prepare_for_export()

    input_example, input_example_length  = get_samples(WAV, total_audio_len)
    input_example = torch.tensor(input_example).to(DEVICE)
    input_example = input_example.repeat(batch_size, 1)
    

    logging.info(f"running {nbatches} batches; with {warmup_batches} batches warmup; batch_size: {batch_size}")
    rtfxs=[]
    for i in range(3): # average over 3 runs
        total_time = 0
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                for i in tqdm(range(nbatches + warmup_batches), desc='Calculating RTF', unit='batch'):
                    for idx in range(0, total_audio_len, chunk_len): # process audio in chunks
                        chunk_singal = input_example[:, idx*SAMPLING_RATE:(idx+chunk_len)*SAMPLING_RATE]
                        chunk_signal_length = torch.tensor(chunk_len * SAMPLING_RATE).to(DEVICE).repeat(batch_size)
                        start = time.time()
                        if decoding_type == 'rnnt':
                            enc_out, enc_len = asr_model.forward(input_signal=chunk_singal, input_signal_length=chunk_signal_length)
                            dec_out, dec_len = asr_model.decoding.rnnt_decoder_predictions_tensor(
                            encoder_output=enc_out, encoded_lengths=enc_len, return_hypotheses=False
                            )
                        elif decoding_type == 'ctc':
                            enc_out, enc_len, greedy_predictions = asr_model.forward(input_signal=chunk_singal, input_signal_length=chunk_signal_length)
                            dec_out, dec_len = asr_model.decoding.ctc_decoder_predictions_tensor(
                            enc_out, decoder_lengths=enc_len, return_hypotheses=False
                            )
                        elif decoding_type == 'aed':
                            log_probs, encoded_len, enc_states, enc_mask = asr_model.forward(input_signal=chunk_singal, input_signal_length=chunk_signal_length)
                            beam_hypotheses = asr_model.decoding.decode_predictions_tensor(
                                encoder_hidden_states=enc_states, 
                                encoder_input_mask=enc_mask, 
                                decoder_input_ids=None, #torch.tensor([[  3,   4,   8,   4,  11]]).to(DEVICE),
                            return_hypotheses=False,
                            )[0]

                            beam_hypotheses = [asr_model.decoding.strip_special_tokens(text) for text in beam_hypotheses]
                        else:
                            raise ValueError(f'Invalid decoding type: {decoding_type}')
                        torch.cuda.synchronize()
                        end = time.time()
                        if i >= warmup_batches:
                            total_time += end - start

        
        avg_time_per_chunk = total_time / total_chunks
        rtf = (avg_time_per_chunk/nbatches) / (chunk_len)
        rtfx = float((1/rtf))
        rtfxs.append(rtfx)
    
    print(f'RTFX: {rtfxs}')
    rtfx = int(sum(rtfxs)/len(rtfxs))
    sys.stdout.write(f'{rtfx}\n')

if __name__ == '__main__':
    main()
