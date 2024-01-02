import os
import json
import pandas
import numpy as np
import nibabel as nib
import torch
import transformers
from transformers import AutoTokenizer
from typing import Dict
import pickle
import sklearn
import math

exclude_files = dict(tunnel=['sub-004','sub-013'], lucy=['sub-053', 'sub-065'])

class Data:
    def __init__(self, tokenizer, task_name):
        with open('/data/gzhch/data/llm_act/{task_name}_1.pkl'.format(task_name=task_name), 'rb') as f:
            llm_act = pickle.load(f)

        with open('/data/gzhch/narratives/stimuli/gentle/{task_name}/align.json'.format(task_name=task_name), 'r') as f:
            raw_input = json.loads(f.read())

        with open('aligned_input.pkl', 'rb') as f:
            all_aligned_inputs = pickle.load(f)
            
        input_ids = tokenizer(raw_input['transcript'])['input_ids']
        
        fmri_files = []
        fmri_imgs = []
        
        folder_path = '/data/gzhch/narratives/derivatives/afni-nosmooth/'

        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith('brain_mask.nii.gz') and f'{task_name}' in filename:
                    self.brain_mask = nib.load(os.path.join(dirpath, filename))

                elif filename.endswith('nii.gz') and f'{task_name}' in filename:
                    participant = filename.split('_')[0]
                    if participant in exclude_files[task_name]:
                        continue
                    file_path = os.path.join(dirpath, filename)
                    fmri_files.append(file_path)
        fmri_files.sort()

        for i, f in enumerate(fmri_files):
            fmri_imgs.append(nib.load(f))

        self.fmri_imgs = fmri_imgs
        self.fmri_files = fmri_files
        self.fmri_act = None
        self.llm_act = llm_act
        self.input_aligned = all_aligned_inputs[task_name]
        self.input_ids = input_ids

        self.tr = 1.5
        self.tr_alignment()

        
    def get_fdata(self):
        fmri_act = [img.get_fdata() for img in self.fmri_imgs]
        max_t = fmri_act[0].shape[-1]
        n = len(fmri_act)
        for i in range(n):
            fmri_act[i] = fmri_act[i].transpose([3,0,1,2]).reshape(max_t, -1)
        self.fmri_act = np.stack(fmri_act)
    
    def tr_alignment(self):
        tr = self.tr

        for i in range(len(self.input_aligned)):
            t = self.input_aligned[-1-i]
            if t['start'] < t['end']:
                max_tr = math.ceil(t['end'] / tr)
                break

        tr_words = [[] for _ in range(max_tr)]
        for c, w in enumerate(self.input_aligned):
            a = math.floor(w['start'] / tr)
            b = math.ceil(w['end'] / tr)
            l, r = w['word_to_token']
            for i in range(a, b):
                tr_words[i].append(c)

        for i in range(len(tr_words)):
            if tr_words[i] == []:
                tr_words[i].append(tr_words[i-1][-1])

        self.tr_to_words = tr_words

        self.tr_to_ids = []
        for words in tr_words:
            l = self.input_aligned[words[0]]['word_to_token'][0]
            r = self.input_aligned[words[-1]]['word_to_token'][1]
            ids = list(range(l, r))
            self.tr_to_ids.append(ids)

    def restore_ffn_gate_and_align(self, layer, align_fn='sum'):
        ## restore topk information back to ffn_gate vectors and then align with the fMRI data
        indices = torch.unbind(self.llm_act['indices'][layer])
        values = torch.unbind(self.llm_act['values'][layer])
        ffn_gate = torch.unbind(torch.zeros(len(indices), 11008).half())
        for i in range(len(indices)):
            ffn_gate[i][indices[i].to(torch.int)] = values[i]
        ffn_gate = torch.stack(ffn_gate)
        tr_words = self.tr_to_words
        tr_llm_act = []
        for ids in self.tr_to_ids:
            if align_fn == 'sum':
                tr_llm_act.append(ffn_gate[ids].sum(dim=0))
            elif align_fn == 'mean':
                tr_llm_act.append(ffn_gate[ids].mean(dim=0))
                
        tr_llm_act = torch.stack(tr_llm_act).float()
        return tr_llm_act
        # tr_llm_act_low_rank, s, _ = torch.pca_lowrank(tr_llm_act, q=100)
