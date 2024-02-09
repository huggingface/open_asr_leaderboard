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
import math
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
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecodingConfig
from nemo.collections.asr.parts.utils.streaming_utils import AudioFeatureIterator, FrameBatchChunkedCTC, FrameBatchChunkedRNNT, FrameBatchMultiTaskAED
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

parser = argparse.ArgumentParser(description='model forward pass profiler / performance tester.')
parser.add_argument("--model", default='nvidia/canary-1b', type=str, help="ASR model")
parser.add_argument("--decoding_type", default='aed', type=str, help="Type of model [rnnt, ctc, aed]")
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
model_stride = 8  # 8 for fastconformer and citrinet, 4 for conformer
chunk_batch_size = 24  # number of chunks to run in parallel
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

def setup_aed_decoding(asr_model):
    decoding_cfg = MultiTaskDecodingConfig()
    decoding_cfg.strategy = "beam"
    decoding_cfg.beam.beam_size = 1
    asr_model.change_decoding_strategy(decoding_cfg)

def setup_rnnt_decoding(asr_model):
    decoding_cfg = RNNTDecodingConfig()
    decoding_cfg.strategy = "greedy_batch"
    if hasattr(asr_model, 'cur_decoder'):
        asr_model.change_decoding_strategy(decoding_cfg, decoder_type="rnnt")
    else:
        asr_model.change_decoding_strategy(decoding_cfg)

def setup_ctc_decoding(asr_model):
    decoding_cfg = CTCDecodingConfig()
    decoding_cfg.strategy = "greedy"
    if hasattr(asr_model, 'cur_decoder'):
        asr_model.change_decoding_strategy(decoding_cfg, decoder_type="ctc")
    else:
        asr_model.change_decoding_strategy(decoding_cfg)

def setup_rnnt_chunk_infer(frame_asr, audio_input):
    frame_reader = AudioFeatureIterator(audio_input, frame_asr.frame_len, frame_asr.raw_preprocessor, frame_asr.asr_model.device)
    frame_asr.set_frame_reader(frame_reader)

def setup_aed_chunk_infer(frame_asr, audio_input, meta_data):
    frame_asr.input_tokens = frame_asr.get_input_tokens(meta_data)
    frame_reader = AudioFeatureIterator(audio_input, frame_asr.frame_len, frame_asr.raw_preprocessor, frame_asr.asr_model.device)
    frame_asr.set_frame_reader(frame_reader)

def setup_ctc_chunk_infer(frame_asr, audio_input):
    frame_reader = AudioFeatureIterator(audio_input, frame_asr.frame_len, frame_asr.raw_preprocessor, frame_asr.asr_model.device)
    frame_asr.set_frame_reader(frame_reader)


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

    frame_asr = None
    if decoding_type == 'aed':
        setup_aed_decoding(asr_model)
        frame_asr = FrameBatchMultiTaskAED(
        asr_model=asr_model,
        frame_len=chunk_len,
        total_buffer=chunk_len,
        batch_size=chunk_batch_size,
    )
    elif decoding_type == 'rnnt':
        setup_rnnt_decoding(asr_model)
        frame_asr = FrameBatchChunkedRNNT(
            asr_model=asr_model,
            frame_len=chunk_len,
            total_buffer=chunk_len,
            batch_size=chunk_batch_size,
        )
    elif decoding_type == 'ctc':
        setup_ctc_decoding(asr_model)
        frame_asr = FrameBatchChunkedCTC(
            asr_model=asr_model,
            frame_len=chunk_len,
            total_buffer=chunk_len,
            batch_size=chunk_batch_size,
        )
    else:
        raise ValueError(f'Invalid decoding type: {decoding_type}, must be one of [ctc, rnnt, aed]')


    logging.info(f"running {nbatches} batches; with {warmup_batches} batches warmup; batch_size: {batch_size}")
    rtfs=[]
    for i in range(3): # average over 3 runs
        total_time = 0
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                for i in tqdm(range(nbatches + warmup_batches)):
                    start = time.time()
                    if decoding_type == 'ctc':
                        frame_asr.reset()
                        setup_ctc_chunk_infer(frame_asr, input_example)
                        hyp = frame_asr.transcribe()
                    elif decoding_type == 'rnnt':
                        frame_asr.reset()
                        setup_rnnt_chunk_infer(frame_asr, input_example)
                        hyp = frame_asr.transcribe()
                    elif decoding_type == 'aed':
                        meta = {
                            'audio_filepath': WAV,
                            'duration': total_audio_len,
                            'source_lang': 'en',
                            'taskname': 'asr',
                            'target_lang': 'en',
                            'pnc': 'yes',
                            'answer': 'nvidia',
                        }
                        frame_asr.reset()
                        setup_aed_chunk_infer(frame_asr, input_example, meta)
                        hyp = frame_asr.transcribe()
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
