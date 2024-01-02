import argparse
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.modeling_llama import LlamaForCausalLM
import time
import pickle
import utils.activation as ana

# ngpu = 1
# heads_per_gpu = 32 //ngpu
# device_map={}
# for gpu in range(ngpu):
#     for l in list(range(0 + (gpu * heads_per_gpu), (0 + (gpu * heads_per_gpu)) + heads_per_gpu)):
#         device_map[f'model.layers.{l}'.format(l=l)] = gpu
# device_map['model.embed_tokens'] = 0
# device_map['lm_head'] = 1
# device_map['model.norm'] = 1

model_dir = '/share/gzhch/resource/models/Llama-2-7b-hf/'
model = LlamaForCausalLM.from_pretrained(
    model_dir,
    device_map='auto',
    torch_dtype=torch.float16,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir)

raw_data = {}
data_dir = '/data/gzhch/narratives/stimuli/transcripts/'
for file in os.listdir(data_dir):
    file_name = file.split('_')[0]
    path = os.path.join(data_dir, file)
    if not path.endswith('txt'):
        continue
    with open(path, 'r') as f:
        raw_data[file_name] = f.read()

name = '21styear'
input_ids = tokenizer(raw_data[name])['input_ids']
input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
input_ids.shape
res = ana.custom_forward(model, input_ids, inspect_acts=['ffn_gate'])

print(res)