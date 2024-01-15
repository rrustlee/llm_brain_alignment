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
parser.add_argument("--roi", type = str, required = True)
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


log_name = f'llama_2_7b_layer_wise_norm-{args.window}-roi_{args.roi}.jsonl'
log_dir = os.path.join(config.RESULT_DIR, log_name)
log_file = open(log_dir, 'w')


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


def layerwise_alignment(stim, resp, s_vox=None, r_vox=None, alphas=None, runs=1, n_train=1000, n_test=500):
    pearson, p = [], []
    for run in range(runs):
        n_total = stim.shape[0]
        ids = random.sample(range(n_total), n_train + n_test)

        if s_vox is not None:
            current_stim = stim[:, s_vox]
        else:
            current_stim = stim

        if r_vox is not None:
            current_resp = resp[:, r_vox]
        else:
            current_resp = resp

        tstim, hstim = current_stim[ids[:n_train]], current_stim[ids[n_train:]]
        tresp, hresp = current_resp[ids[:n_train]], current_resp[ids[n_train:]]

        if alphas is None:
            alphas = torch.tensor([100 for _ in range(current_resp.shape[-1])]).cuda()

        elif alphas == 'adaptive':
            nchunks = int(np.ceil(tresp.shape[0] / 5 / 100))
            weights, alphas, bscorrs = bootstrap_ridge_torch(tstim, tresp, use_corr = False, alphas = np.logspace(0, 3, 10),
                    nboots = 3, chunklen = 100, nchunks = nchunks)        

        bs_weights = ridge_torch(tstim, tresp, alphas)
        bs_weights = bs_weights.to(hstim.device).to(hstim.dtype)
        pred = hstim.matmul(bs_weights)
        pred = pred.cpu().numpy() 
        hresp = hresp.cpu().numpy()
        stat = scipy.stats.pearsonr(pred.flatten(), hresp.flatten())
        pearson.append(stat.statistic)
        p.append(stat.pvalue)
    return pearson, p
    # return torch.tensor(pearson).mean().item(), torch.tensor(pearson).std().item(), torch.tensor(p).mean().item(), torch.tensor(p).std().item()


args2 = copy.deepcopy(args)

layer1 = 10
layer2 = 20

args.layer = layer1

for layer1 in range(0, 31, 2):
# for layer1 in [10]:
    rstim, r_mean, r_std = get_stim_torch(args, stories, llama)
    rstim_norm = normalize(rstim, r_mean, r_std)

    for layer2 in range(0, 31, 2):
        args2.layer = layer2
        rresp, r_mean, r_std = get_stim_torch(args2, stories, llama)
        rresp_norm = normalize(rresp, r_mean, r_std)

        if args.roi.startswith('mean_least'):
            n_vox = int(args.roi.split(':')[1])
            r_vox = r_mean.topk(n_vox, largest=False).indices
        elif args.roi.startswith('mean_most'):
            n_vox = int(args.roi.split(':')[1])
            r_vox = r_mean.topk(n_vox, largest=True).indices
        elif args.roi.startswith('mean_th_most'):
            th = float(args.roi.split(':')[1])
            r_vox = torch.nonzero(r_mean>th*r_mean.max())[:, 0]
            n_vox = len(r_vox)
        elif args.roi.startswith('std_least'):
            n_vox = int(args.roi.split(':')[1])
            r_vox = r_std.topk(n_vox, largest=False).indices
        elif args.roi.startswith('std_most'):
            n_vox = int(args.roi.split(':')[1])
            r_vox = r_std.topk(n_vox, largest=True).indices
        elif args.roi.startswith('std_th_most'):
            th = float(args.roi.split(':')[1])
            r_vox = torch.nonzero(r_std>th*r_std.max())[:, 0]
            n_vox = len(r_vox)
        elif args.roi == 'all':
            r_vox = None
            n_vox = 'all'

        pearson, p = layerwise_alignment(rstim, rresp, s_vox=None, r_vox=r_vox, alphas='adaptive', runs=args.runs)
        pearson_norm, p_norm = layerwise_alignment(rstim_norm, rresp_norm, s_vox=None, r_vox=r_vox, alphas='adaptive', runs=args.runs)
        
        res_args = dict(roi=args.roi, 
                        layer1=layer1, 
                        layer2=layer2, 
                        window=args.window, 
                        n_vox=n_vox, 
                        pearson=pearson, 
                        p=p, 
                        peason_mean=torch.tensor(pearson).mean().item(), 
                        pearson_norm=pearson_norm, 
                        p_norm=p_norm, 
                        peason_norm_mean=torch.tensor(pearson_norm).mean().item())

        print(layer1, layer2, n_vox, torch.tensor(pearson).mean().item(), torch.tensor(pearson_norm).mean().item())
        json.dump(res_args, log_file)
        log_file.write('\n')
