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

Example:
    python calc_rtf.py --model stt_en_conformer_ctc_large --decoding_type ctc
"""
import copy 
from omegaconf import OmegaConf
import time
import argparse
from tqdm import tqdm
import torch
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
parser.add_argument("--audio", default="../data/sample_4469669.wav", type=str, help="wav file to use")

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


def extract_preprocessor(model, device):
    cfg = copy.deepcopy(model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    preprocessor = model.from_config_dict(cfg.preprocessor)
    return preprocessor.to(device)

def main():

    if MODEL.endswith('.nemo'):
        asr_model = ASRModel.restore_from(MODEL)
    else:
        asr_model = ASRModel.from_pretrained(MODEL)

    asr_model.to(DEVICE)
    asr_model.eval()
    asr_model._prepare_for_export()

    preprocessor = extract_preprocessor(asr_model, DEVICE)
    input_example, input_example_length  = get_samples(WAV, total_audio_len)
    input_example = torch.tensor(input_example).to(DEVICE)
    input_example = input_example.repeat(batch_size, 1)
    input_example_length = torch.tensor(input_example_length).to(DEVICE)
    input_example_length = input_example_length.repeat(batch_size)

    processed_signal, processed_signal_length = preprocessor(input_signal=input_example, length=input_example_length)
    processed_example = processed_signal.repeat(batch_size, 1, 1)
    

    logging.info(f"running {nbatches} batches; with {warmup_batches} batches warmup; batch_size: {batch_size}")
    rtfs=[]
    for i in range(3): # average over 3 runs
        total_time = 0
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                for i in tqdm(range(nbatches + warmup_batches)):
                    start = time.time()
                    if decoding_type == 'rnnt':
                        enc_out, enc_len = asr_model.encoder.forward(audio_signal=processed_example, length=processed_signal_length)
                        dec_out, dec_len = asr_model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=enc_out, encoded_lengths=enc_len, return_hypotheses=False
                        )
                    elif decoding_type == 'ctc':
                        enc_out, enc_len, greedy_predictions = asr_model.forward(input_signal=input_example, input_signal_length=input_example_length)
                        dec_out, dec_len = asr_model.decoding.ctc_decoder_predictions_tensor(
                        enc_out, decoder_lengths=enc_len, return_hypotheses=False
                        )
                    elif decoding_type == 'aed':
                        log_probs, encoded_len, enc_states, enc_mask = asr_model.forward(input_signal=input_example, input_signal_length=input_example_length)
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

    
        rtf = (total_time/nbatches) / (float(input_example_length) / 16000)
        
        rtfs.append(rtf)
    
    print(f'RTF: {rtfs}')
    rtf = sum(rtfs)/len(rtfs)
    sys.stdout.write(f'{rtf:.4f}\n')

if __name__ == '__main__':
    main()
