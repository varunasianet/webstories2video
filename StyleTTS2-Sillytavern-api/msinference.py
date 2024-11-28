from cached_path import cached_path
import nltk
nltk.download('punkt')
from scipy.io.wavfile import write
import torch
import yaml
from torch.cuda.amp import autocast, GradScaler
from scipy.io.wavfile import write
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np

from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)  # Assign the result to mel_tensor

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


import yaml

# Load GPU config from file
with open('gpu_config.yml', 'r') as file:
    gpu_config = yaml.safe_load(file)

# Extract GPU device ID from config
gpu_device_id = gpu_config.get('gpu_device_id', 0)

# Check if CUDA is available
if torch.cuda.is_available() and gpu_device_id != 999:
    # Set the device to the specified GPU
    torch.cuda.set_device(gpu_device_id)
    device = torch.device('cuda')
else:
    # If CUDA is not available or GPU ID is 999, use CPU
    device = torch.device('cpu')

#print(f"Selected device: {device}")

# Enable mixed precision training
# Enable mixed precision training
use_amp = True
scaler = GradScaler(enabled=use_amp)

# Reduce batch size
batch_size = 1  # Use a smaller batch size

# Implement gradient accumulation
accumulation_steps = 4

# Clear CUDA cache
torch.cuda.empty_cache()

# Function to monitor GPU memory usage
def monitor_gpu_memory():
    print("GPU Memory Summary:")
    print(torch.cuda.memory_summary())

import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
# phonemizer = Phonemizer.from_checkpoint(str(cached_path('https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt')))


# config = yaml.safe_load(open("Models/LibriTTS/config.yml"))
config = yaml.safe_load(open(str(cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/config.yml"))))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
params_whole = torch.load(str(cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth")), map_location='cpu')
params = params_whole['net']

# Move parameters to the desired device
def recursive_to(device, nested_dict):
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            nested_dict[key] = recursive_to(device, value)
        else:
            nested_dict[key] = value.to(device)
    return nested_dict

# Move all tensors in the nested OrderedDict to the desired device
params = recursive_to(device, params)


for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

import re
from num2words import num2words
from datetime import datetime
import re
from num2words import num2words
from datetime import datetime
import re
from num2words import num2words
from datetime import datetime

def indian_currency_to_words(amount):
    amount = amount.replace(',', '')
    rupees = int(float(amount))
    paise = int((float(amount) - rupees) * 100)
    
    words = ['rupees']
    if rupees:
        if rupees >= 10000000:
            crores = rupees // 10000000
            rupees %= 10000000
            words.append(f"{num2words(crores)} crore")
        if rupees >= 100000:
            lakhs = rupees // 100000
            rupees %= 100000
            words.append(f"{num2words(lakhs)} lakh")
        if rupees >= 1000:
            thousands = rupees // 1000
            rupees %= 1000
            words.append(f"{num2words(thousands)} thousand")
        if rupees > 0:
            words.append(num2words(rupees))
    else:
        words.append("zero")
    
    if paise:
        words.extend(["and", num2words(paise), "paise"])
    return " ".join(words)

def preprocess_text(text):
    # Handle currency with "/gm"
    text = re.sub(r'(Rs\.?|₹)\s?(\d+(?:,\d+)*(?:\.\d+)?)/gm', 
                  lambda m: f"rupees {indian_currency_to_words(m.group(2))} per gram", text)
    
    # Handle currency without "/gm"
    text = re.sub(r'(Rs\.?|₹)\s?(\d+(?:,\d+)*(?:\.\d+)?)', 
                  lambda m: f"rupees {indian_currency_to_words(m.group(2))}", text)
    
    # Remove special characters between words, numbers, and alphanumeric characters
    text = re.sub(r'(\w)([^\w\s.,;:?!-])(\w)', r'\1 \3', text)
    
    # Handle dates
    text = re.sub(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b', 
                  lambda m: f"{num2words(int(m.group(1)), to='ordinal')} {datetime.strptime(m.group(2), '%m').strftime('%B')} {num2words(int(m.group(3)))}", text)
    
    # Handle time
    text = re.sub(r'\b(\d{1,2}):(\d{2})\b', 
                  lambda m: f"{num2words(int(m.group(1)))} {num2words(int(m.group(2))) if int(m.group(2)) else 'hundred'}", text)
    
    # Handle percentages
    text = re.sub(r'(\d+(?:\.\d+)?)%', lambda m: f"{num2words(float(m.group(1)))} percent", text)
    
    # Handle other numbers
    text = re.sub(r'\b(\d+(?:,\d+)*(?:\.\d+)?)\b', lambda m: num2words(float(m.group(1).replace(',', ''))), text)
    
    # Handle abbreviations and special cases
    abbrev_map = {
        'Mr.': 'Mister', 'Mrs.': 'Misses', 'Dr.': 'Doctor', 'St.': 'Saint',
        'Govt.': 'Government', 'dept.': 'department', 'vs.': 'versus',
        'etc.': 'et cetera', 'i.e.': 'that is', 'e.g.': 'for example',
    }
    for abbrev, expansion in abbrev_map.items():
        text = text.replace(abbrev, expansion)
    
    # Handle hyphens between words
    text = re.sub(r'(\w+)-(\w+)', r'\1 \2', text)
    
    return text
    
def date_to_words(date_string):
    try:
        date = datetime.strptime(date_string.replace('/', '-'), '%d-%m-%Y')
        return date.strftime('%d %B %Y')
    except ValueError:
        return date_string

def time_to_words(hour, minute):
    hour = int(hour)
    minute = int(minute)
    return f"{num2words(hour)} {num2words(minute) if minute else 'hundred'}"

def number_to_words(number):
    number = number.replace(',', '')
    if '.' in number:
        integer_part, decimal_part = number.split('.')
        return f"{num2words(int(integer_part))} point {' '.join(num2words(int(d)) for d in decimal_part)}"
    else:
        return num2words(int(number))
        
def inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1, speed=1.0, use_gruut=False):
    # Preprocess the text
    text = preprocess_text(text.strip())
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    
    # Create or load your tokens tensor
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        
        print('inference settings')
        print(f"alpha: {alpha}")
        print(f"beta: {beta}")
        print(f"steps: {diffusion_steps}")
        print(f"scale: {embedding_scale}")
        print(f"speed: {speed}")

        # Use mixed precision to reduce memory usage
        with autocast(enabled=use_amp):
            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                             embedding=bert_dur,
                             embedding_scale=embedding_scale,
                             features=ref_s,
                             num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    # Clear unused tensors and cache
    del tokens, t_en, bert_dur, d_en, s_pred, s, ref, d, x, duration, pred_dur, pred_aln_trg, en, F0_pred, N_pred, asr, asr_new
    torch.cuda.empty_cache()
    gc.collect()

    # Monitor GPU memory usage
    monitor_gpu_memory()

    return out.squeeze().cpu().numpy()[..., :-50]

def LFinference(text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1, use_gruut=False):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    ps = ps.replace('``', '"')
    ps = ps.replace("''", '"')

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                        embedding=bert_dur,
                                        embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                            num_steps=diffusion_steps).squeeze(1)

        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en,
                                        s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later

def STinference(text, ref_s, ref_text, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, use_gruut=False):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    ref_text = ref_text.strip()
    ps = global_phonemizer.phonemize([ref_text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    ref_tokens = textclenaer(ps)
    ref_tokens.insert(0, 0)
    ref_tokens = torch.LongTensor(ref_tokens).to(device).unsqueeze(0)


    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(device)
        ref_text_mask = length_to_mask(ref_input_lengths).to(device)
        ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later


torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)  # Assign the result to mel_tensor

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


import yaml

# Load GPU config from file
with open('gpu_config.yml', 'r') as file:
    gpu_config = yaml.safe_load(file)

# Extract GPU device ID from config
gpu_device_id = gpu_config.get('gpu_device_id', 0)

# Check if CUDA is available
if torch.cuda.is_available() and gpu_device_id != 999:
    # Set the device to the specified GPU
    torch.cuda.set_device(gpu_device_id)
    device = torch.device('cuda')
else:
    # If CUDA is not available or GPU ID is 999, use CPU
    device = torch.device('cpu')

#print(f"Selected device: {device}")


import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
# phonemizer = Phonemizer.from_checkpoint(str(cached_path('https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt')))


# config = yaml.safe_load(open("Models/LibriTTS/config.yml"))
config = yaml.safe_load(open(str(cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/config.yml"))))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
params_whole = torch.load(str(cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth")), map_location='cpu')
params = params_whole['net']

# Move parameters to the desired device
def recursive_to(device, nested_dict):
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            nested_dict[key] = recursive_to(device, value)
        else:
            nested_dict[key] = value.to(device)
    return nested_dict

# Move all tensors in the nested OrderedDict to the desired device
params = recursive_to(device, params)


for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

import re
from num2words import num2words

def preprocess_text(text):
    # Handle Indian currency
    text = re.sub(r'(Rs|₹)\s?(\d+(?:,\d+)*(?:\.\d+)?)', lambda m: indian_currency_to_words(m.group(2)), text)
    
    # Handle hyphens
    text = re.sub(r'(\w+)-(\w+)', r'\1 \2', text)
    
    # Handle abbreviations
    abbrev_map = {
        'Mr.': 'Mister',
        'Mrs.': 'Misses',
        'Dr.': 'Doctor',
        'St.': 'Saint',
        'Govt.': 'Government',
        'dept.': 'department',
    }
    for abbrev, expansion in abbrev_map.items():
        text = text.replace(abbrev, expansion)
    
    # Handle dates
    text = re.sub(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b', lambda m: date_to_words(m.group(1)), text)
    
    # Handle numbers
    text = re.sub(r'\b\d+(?:\.\d+)?\b', lambda m: num2words(float(m.group())), text)
    
    # Handle ordinals
    text = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', lambda m: num2words(int(m.group(1)), to='ordinal'), text)
    
    return text

def indian_currency_to_words(amount):
    amount = amount.replace(',', '')
    rupees, *paise = amount.split('.')
    rupees = int(rupees)
    paise = int(paise[0]) if paise else 0
    
    words = []
    if rupees:
        if rupees >= 10000000:
            crores = rupees // 10000000
            rupees %= 10000000
            words.append(f"{num2words(crores)} crore")
        if rupees >= 100000:
            lakhs = rupees // 100000
            rupees %= 100000
            words.append(f"{num2words(lakhs)} lakh")
        if rupees:
            words.append(num2words(rupees))
        words.append("rupees")
    if paise:
        words.extend([num2words(paise), "paise"])
    return " ".join(words)

def date_to_words(date_string):
    try:
        date = datetime.strptime(date_string.replace('/', '-'), '%d-%m-%Y')
        return date.strftime('%d %B %Y')
    except ValueError:
        return date_string

def inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1, speed=1.0, use_gruut=False):
     # Preprocess the text
    text = preprocess_text(text.strip())
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    
    device = torch.device(f"cuda:{gpu_device_id}" if torch.cuda.is_available() else "cpu")

    # Create or load your tokens tensor
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        
        print('inference settings')
        print(f"alpha: {alpha}")
        print(f"beta: {beta}")
        print(f"steps: {diffusion_steps}")
        print(f"scale: {embedding_scale}")
        print(f"speed: {speed}")

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

def LFinference(text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1, use_gruut=False):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    ps = ps.replace('``', '"')
    ps = ps.replace("''", '"')

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                        embedding=bert_dur,
                                        embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                            num_steps=diffusion_steps).squeeze(1)

        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = t * s_prev + (1 - t) * s_pred

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en,
                                        s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later

def STinference(text, ref_s, ref_text, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, use_gruut=False):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    ref_text = ref_text.strip()
    ps = global_phonemizer.phonemize([ref_text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    ref_tokens = textclenaer(ps)
    ref_tokens.insert(0, 0)
    ref_tokens = torch.LongTensor(ref_tokens).to(device).unsqueeze(0)


    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(device)
        ref_text_mask = length_to_mask(ref_input_lengths).to(device)
        ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
