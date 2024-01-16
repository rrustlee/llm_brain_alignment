import os
import copy
import numpy as np
import json
import argparse
import random
import scipy
import config
from GPT import GPT
from LLAMA import LLAMA
from StimulusModel import LMFeatures
from utils_stim import get_story_wordseqs
# from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge, ridge_corr
from utils_ridge.ridge_torch import ridge_torch, bootstrap_ridge_torch, ridge_corr_torch
from utils_ridge.stimulus_utils import TRFile, load_textgrids, load_simulated_trfiles
from utils_ridge.dsutils import make_word_ds
from utils_ridge.interpdata import lanczosinterp2D, lanczosinterp2D_torch
from utils_ridge.util import make_delayed
from utils_ridge.utils import mult_diag, counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import utils_llama.activation as ana

import scipy
import math
import matplotlib.pyplot as plt

import time
import h5py

random.seed(42)

class ARGS:
    def __init__(self):
        self.subject = 'S1'
        self.gpt = 'perceived'
        self.sessions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
        self.layer = 20
        self.act_name = 'ffn_gate'

args = ARGS()

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type = str, required = False)
parser.add_argument("--gpt", type = str, default = "perceived")
parser.add_argument("--sessions", nargs = "+", type = int, 
    default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
parser.add_argument("--act_name", type = str, default = "ffn_gate")
parser.add_argument("--window", type = int, required = True)
parser.add_argument("--layer", type = int)
parser.add_argument("--runs", type = int, default = 3)
parser.add_argument("--chunk", type = int, default = 2)

args = parser.parse_args()

# torch.cuda.memory._record_memory_history()
torch.cuda.empty_cache()
torch.set_grad_enabled(False)

# training stories
stories = []
with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
    sess_to_story = json.load(f) 
for sess in args.sessions:
    stories.extend(sess_to_story[str(sess)])

stories = stories[:10]

model_dir = '/ossfs/workspace/nas/gzhch/data/models/Llama-2-7b-hf'
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir, 
#     device_map='auto',
#     torch_dtype=torch.float16,
# ).eval()

model = None

tokenizer = AutoTokenizer.from_pretrained(model_dir)


## load cached llm act if possible
cache_dir = '/ossfs/workspace/act_cache'
llama = LLAMA(model, tokenizer, cache_dir)


log_name = f'llama_2_7b_layer_wise-{args.window}.jsonl'
log_name_norm = f'llama_2_7b_layer_wise_norm-{args.window}.jsonl'
log_dir = os.path.join(config.RESULT_DIR, log_name)
log_dir_norm = os.path.join(config.RESULT_DIR, log_name_norm)
log_file = open(log_dir, 'w')
log_file_norm = open(log_dir_norm, 'w')


def get_stim_torch(args, stories, llama):
    word_seqs = get_story_wordseqs(stories)
    word_vecs = {}
    for story in stories:
        words = word_seqs[story].data
        embs = llama.get_llm_act(story, words, args.window, args.act_name, args.layer, chunk=args.chunk, cache_all_layer=False)
        word_vecs[story] = embs
    ds_mat = torch.vstack([torch.tensor(word_vecs[story]) for story in stories]).float().cuda()
    r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
    r_std[r_std == 0] = 1
    
    return ds_mat, r_mean, r_std

def normalize(ds_mat, r_mean, r_std):
    return torch.nan_to_num(torch.matmul((ds_mat - r_mean), torch.linalg.inv(torch.diag(r_std))))

def layerwise_alignment(stim, resp, rois, alphas='adaptive', runs=1, n_train=1000, n_test=500):
    r_mean, r_std = resp.mean(0), resp.std(0)
    r_std[r_std == 0] = 1

    result = {}
    for roi in rois:
        for stat in ['std', 'weighted_std', 'pearson', 'p']:
            result[f'{roi}-{stat}'] = []

    for run in range(runs):
        n_total = stim.shape[0]
        ids = random.sample(range(n_total), n_train + n_test)

        tstim, hstim = stim[ids[:n_train]], stim[ids[n_train:]]
        tresp, hresp = resp[ids[:n_train]], resp[ids[n_train:]]

        if alphas is None:
            alphas = torch.tensor([1 for _ in range(resp.shape[-1])]).cuda()

        elif alphas == 'adaptive':
            nchunks = int(np.ceil(tresp.shape[0] / 5 / 100))
            weights, alphas, bscorrs = bootstrap_ridge_torch(tstim, tresp, use_corr = False, alphas = np.logspace(0, 3, 10),
                    nboots = 3, chunklen = 100, nchunks = nchunks)        

        bs_weights = ridge_torch(tstim, tresp, alphas)
        bs_weights = bs_weights.to(hstim.device).to(hstim.dtype)
        pred = hstim.matmul(bs_weights)

        for roi in rois:
            if roi.startswith('mean_least'):
                n_vox = int(roi.split(':')[1])
                r_vox = r_mean.topk(n_vox, largest=False).indices
            elif roi.startswith('mean_most'):
                n_vox = int(roi.split(':')[1])
                r_vox = r_mean.topk(n_vox, largest=True).indices
            elif roi.startswith('mean_th_most'):
                th = float(roi.split(':')[1])
                r_vox = torch.nonzero(r_mean>th*r_mean.max())[:, 0]
                n_vox = len(r_vox)
            elif roi.startswith('std_least'):
                n_vox = int(roi.split(':')[1])
                r_vox = r_std.topk(n_vox, largest=False).indices
            elif roi.startswith('std_most'):
                n_vox = int(roi.split(':')[1])
                r_vox = r_std.topk(n_vox, largest=True).indices
            elif roi.startswith('std_th_most'):
                th = float(roi.split(':')[1])
                r_vox = torch.nonzero(r_std>th*r_std.max())[:, 0]
                n_vox = len(r_vox)
            elif roi == 'all':
                r_vox = None
                n_vox = len(r_std)

            if r_vox is not None:
                x, y = pred[:, r_vox], hresp[:, r_vox]
            else:
                x, y = pred, hresp

            res = calculate_diff(x, y)
            result[f'{roi}-n_vox'] = n_vox
            for stat in ['std', 'weighted_std', 'pearson', 'p']:
                result[f'{roi}-{stat}'].append(res[stat])

        for roi in rois:
            for stat in ['std', 'weighted_std', 'pearson', 'p']:
                result[f'{roi}-{stat}-mean'] = torch.tensor(result[f'{roi}-{stat}']).mean().item()

    return result

def calculate_diff(x, y):
    std = (x-y).std().item()
    weighted_std = ((x-y)*y).std().item()
    pearson = scipy.stats.pearsonr(x.reshape(-1).cpu(), y.reshape(-1).cpu())
    return dict(std=std, weighted_std=weighted_std, pearson=pearson.statistic, p = pearson.pvalue)


args2 = copy.deepcopy(args)

for layer1 in range(0, 31, 2):
# for layer1 in [10]:
    args.layer = layer1
    rstim, r_mean, r_std = get_stim_torch(args, stories, llama)
    rstim_norm = normalize(rstim, r_mean, r_std)

    for layer2 in range(0, 31, 2):
        args2.layer = layer2
        rresp, r_mean, r_std = get_stim_torch(args2, stories, llama)
        rresp_norm = normalize(rresp, r_mean, r_std)

        rois = ['all', 
                'mean_most:1000', 
                'mean_least:1000',
                'mean_th_most:0.1', 
                'std_most:1000',
                'std_least:1000',
                'std_th_most:0.2']

        res = layerwise_alignment(rstim, rresp, rois, alphas='adaptive', runs=args.runs, n_train=1000, n_test=1000)
        res_norm = layerwise_alignment(rstim_norm, rresp_norm, rois, alphas='adaptive', runs=args.runs, n_train=1000, n_test=1000)
        res.update(dict(layer1=layer1, layer2=layer2, window=args.window))
        res_norm.update(dict(layer1=layer1, layer2=layer2, window=args.window))
    
        json.dump(res, log_file)
        log_file.write('\n')
        log_file.flush()
        json.dump(res_norm, log_file_norm)
        log_file_norm.write('\n')
        log_file_norm.flush()

        print(layer1, layer2, res['all-pearson-mean'], res_norm['all-pearson-mean'])