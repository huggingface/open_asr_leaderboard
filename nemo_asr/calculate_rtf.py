"""
NeMo ASR Model Profiler

This script performs a forward pass on an NeMo ASR models and measures its real-time factor (RTF).
RTF is a metric used to evaluate the processing speed of ASR models.

Parameters:
    --model: ASR model name or path to the model checkpoint file.
    --decoding_type: Type of decoding to use (ctc or rnnt).
    --gpu: GPU device to use.
    --batch_size: Batch size to use for inference.
    --nbatches: Total number of batches to process.
    --warmup_batches: Number of batches to skip as warmup.
    --audio: Path to the input audio file for ASR.
    --audio_maxlen: Maximum duration of audio to process (in seconds).
    --precision: Model precision (16, 32, or bf16).
    --cudnn_benchmark: Enable cuDNN benchmarking.
    --log: Enable logging.

Example:
    python calculate_rtf.py --model stt_en_conformer_ctc_large --decoding_type ctc --gpu 0 --batch_size 1 --nbatches 5 --warmup_batches 5 --audio /path/to/audio.wav --audio_maxlen 600 --precision bf16 --cudnn_benchmark
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

from nemo.utils import logging
from contextlib import nullcontext
from nemo.collections.asr.models import ASRModel



parser = argparse.ArgumentParser(description='model forward pass profiler / performance tester.')
parser.add_argument("--model", default='stt_en_conformer_ctc_large', type=str, help="ASR model")
parser.add_argument("--decoding_type", default='ctc', type=str, help="Encoding type (bpe or char)")
parser.add_argument("--gpu", default=0, type=int, help="GPU device to use")
parser.add_argument("--batch_size", default=1, type=int, help="batch size to use")
parser.add_argument("--nbatches", default=5, type=int, help="Total Number of batches to process")
parser.add_argument("--warmup_batches", default=5, type=int, help="Number of batches to skip as warmup")
parser.add_argument("--audio", default="/disk3/datasets/speech-datasets/earnings22/media/4469669.wav", type=str, help="wav file to use")
parser.add_argument("--audio_maxlen", default=16, type=float, help="cut the file at given length if it is longer")
parser.add_argument("--precision", default='bf16', type=str, help="precision: 16/32/bf16")
parser.add_argument("--cudnn_benchmark", dest="enable_cudnn_bench", action="store_true", help="toggle cudnn benchmarking", default=True)
parser.add_argument("--log", dest="log", action="store_true", help="toggle logging", default=True)

args = parser.parse_args()

if args.log:
    # INFO
    logging.setLevel(20)
else:
    logging.setLevel(0)

if args.enable_cudnn_bench:
    torch.backends.cudnn.benchmark=True

PRECISION = args.precision
WAV = args.audio
audio_maxlen = args.audio_maxlen
MODEL = args.model
batch_size = args.batch_size
nbatches = args.nbatches
warmup_batches = args.warmup_batches
decoding_type = args.decoding_type

DEVICE=torch.device(args.gpu)

if PRECISION != 'bf16' and PRECISION != '16' and PRECISION != '32':
    logging.error(f'unknown precision: {PRECISION}')
    sys.exit(1)

logging.info(f'precision: {PRECISION}')
logging.info(f'WAV: {WAV}')
logging.info(f'AUDIO MAXLEN: {audio_maxlen}')
logging.info(f'MODEL: {MODEL}')
logging.info(f'batch_size: {batch_size}')
logging.info(f'num batches: {nbatches}')
logging.info(f'cudnn_benchmark: {args.enable_cudnn_bench}')


def get_samples(audio_file, audio_maxlen, target_sr=16000):
    with sf.SoundFile(audio_file, 'r') as f:
        dtype = 'int16'
        sample_rate = f.samplerate
        samples = f.read(dtype=dtype)
        if sample_rate != target_sr:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
        samples = samples.astype('float32') / 32768
        samples = samples.transpose()
        sample_length = samples.shape[0]
        if sample_length > audio_maxlen * target_sr:
            logging.info(f'resizing audio sample from {sample_length / target_sr} to maxlen of {audio_maxlen}')
            sample_length = int(audio_maxlen * target_sr)
            samples = samples[:sample_length]
            logging.info(f'new sample lengh: {samples.shape[0]}')
        else:
            pad_length = int(audio_maxlen * target_sr) - sample_length
            logging.info(f'padding audio sample from {sample_length / target_sr} to maxlen of {audio_maxlen}')
            samples = np.pad(samples, (0, pad_length), 'constant', constant_values=(0, 0))
            sample_length = int(audio_maxlen * target_sr)
            
        return samples, sample_length

def preprocess_audio(preprocessor, audio, device):
    audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)

    audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
    processed_signal, processed_signal_length = preprocessor(
        input_signal=audio_signal, length=audio_signal_len
    )
    return processed_signal, processed_signal_length

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
    asr_model.encoder.eval()
    asr_model.encoder.freeze()
    asr_model._prepare_for_export()

    processor = extract_preprocessor(asr_model, DEVICE)

    input_example, input_example_length  = get_samples(WAV, audio_maxlen)
    logging.info(f'processed example shape: {input_example.shape}')
    logging.info(f'processed example length shape: {input_example_length}')
    processed_example, processed_example_length = preprocess_audio(processor, input_example, DEVICE)
    processed_example = processed_example.repeat(batch_size, 1, 1)
    processed_example_length = processed_example_length.repeat(batch_size)
    logging.info(f'processed example shape: {processed_example.size()}')
    logging.info(f'processed example length shape: {processed_example_length.size()}')

    profiling_context = nullcontext()
    # if FP16:
    if PRECISION == '16':
        precision_context = torch.cuda.amp.autocast()
    elif PRECISION == 'bf16':
        precision_context = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif PRECISION == '32':
        pass
    else:
        logging.error(f'unknown precision: {PRECISION}')
        sys.exit(1)


    if decoding_type == 'ctc':
        asr_model.change_decoding_strategy(decoding_cfg=None)
    
    logging.info(f"running {nbatches} batches; with {warmup_batches} batches warmup; batch_size: {batch_size}")
    rtfs=[]
    for i in range(3): # average over 3 runs
        total_time = 0
        with profiling_context:
            with precision_context:
                with torch.no_grad():
                    for i in tqdm(range(nbatches + warmup_batches)):

                        start = time.time()
                        if decoding_type == 'rnnt':
                            enc_out, enc_len = asr_model.encoder.forward(audio_signal=processed_example, length=processed_example_length)
                            dec_out, dec_len = asr_model.decoding.rnnt_decoder_predictions_tensor(
                            encoder_output=enc_out, encoded_lengths=enc_len, return_hypotheses=False
                            )
                        else:
                            enc_out, enc_len, greedy_predictions = asr_model.forward(processed_signal=processed_example, processed_signal_length=processed_example_length)
                            dec_out, dec_len = asr_model.decoding.ctc_decoder_predictions_tensor(
                            enc_out, decoder_lengths=enc_len, return_hypotheses=False
                            )
                        torch.cuda.synchronize()
                        end = time.time()
                        if i >= warmup_batches:
                            total_time += end - start

    
        rtf = (total_time/nbatches) / (input_example_length / 16000)
        
        rtfs.append(rtf)
    
    print(f'RTF: {rtfs}')
    rtf = sum(rtfs)/len(rtfs)
    sys.stdout.write(f'{rtf:.4f}\n')

if __name__ == '__main__':
    main()
