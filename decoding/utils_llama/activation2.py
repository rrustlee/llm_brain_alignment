import torch
import os
import gc
import torch.nn as nn
from torch.nn.functional import pad
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import random
from functools import partial
# from custom.modeling_llama import LlamaForCausalLM
from datasets.utils.logging import disable_progress_bar
import datasets
from typing import List, Mapping, NewType, Optional, Tuple, Union
import pickle
from collections import Counter
import math
import time

disable_progress_bar()


# logging.disable_progress_bar()

# import utils.mmlu as mmlu


@torch.no_grad()
def custom_forward(model,
                   input_ids,
                   position_ids=None,
                   past_key_values=None,
                   early_exit=None,
                   attention_mask=None,
                   inspect_acts=[],
                   forward_layer_ids=None,
                   skip_layer_ids=None,
                   head_mask=None,
                   record_layer_ids=None,
                   return_logits=True,
                   add_attn=True,
                   add_ffn=True,
                   fake_act_args=None,
                   use_cache=True,
                   draft_config=None,
                   output_attentions=None,
                   debug=False,
                   output_hidden_states=None,
                   token_type_ids=None,
                   inputs_embeds=None,
                   return_dict=None
                   ):
    ### inspect_acts : ffn, attn, ffn_gate
    # forward_layer_ids : layers that are selected as the small model
    # record_layer_ids : layers that directly contribute to the final logits

    # assert forward_layer_ids is None or skip_layer_ids is None
    # output_attentions = output_attentions if output_attentions is not None else model.transformer.config.output_attentions
    # # support attn ffn ffn_gate layer_input
    # activations = {i: [] for i in inspect_acts}
    # input_shape = input_ids.size()
    # input_ids = input_ids.view(-1, input_shape[-1])
    # # batch_size = input_ids.shape[0]
    # # batch_size, seq_length = input_ids.shape
    # # seq_length_with_past = seq_length
    # device = input_ids.device
    # past_length = 0
    # past_key_values = tuple([None] * len(model.transformer.h))

    # position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    # position_ids = position_ids.unsqueeze(0)

    # head_mask = model.transformer.get_head_mask(head_mask, model.transformer.config.n_layer)
    # inputs_embeds = model.transformer.wte(input_ids)


    # hidden_states = inputs_embeds
    # hidden_states = model.transformer.drop(hidden_states)




    # for idx in list(range(len(model.transformer.h))):

    #     decoder_layer = model.transformer.h[idx]
    #     layer_past = past_key_values[idx] if past_key_values is not None else None

    #     #################################
    #     ### customized self attention ###
    #     #################################  

    #     if 'ffn_gate' in activations.keys():
    #         gates= decoder_layer.mlp.act(decoder_layer.mlp.fc_in(decoder_layer.ln_1(hidden_states)))
    #         # gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states))
    #         # hidden_states = decoder_layer.mlp.down_proj(gates * decoder_layer.mlp.up_proj(hidden_states))
    #         activations['ffn_gate'].append(gates.cpu())
    #         hidden_states = decoder_layer.mlp(hidden_states)
    #         del gates
    #         torch.cuda.empty_cache()
    #     outputs = decoder_layer(
    #                 hidden_states=hidden_states,
    #                 layer_past=layer_past,
    #                 attention_mask=attention_mask,
    #                 position_ids=position_ids,
    #                 head_mask=head_mask[idx],
    #                 use_cache=use_cache,
    #                 output_attentions=output_attentions,
    #             )
    #     hidden_states = outputs[0]
    activations = {i: [] for i in inspect_acts}    
    output_attentions = output_attentions if output_attentions is not None else model.transformer.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model.transformer.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else model.transformer.config.use_cache
    return_dict = return_dict if return_dict is not None else model.transformer.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        model.transformer.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(model.transformer.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

    # Attention mask.
    if attention_mask is not None:
        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")
        attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(dtype=model.transformer.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(model.transformer.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
    head_mask = model.transformer.get_head_mask(head_mask, model.transformer.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = model.transformer.wte(input_ids)

    hidden_states = inputs_embeds

    if token_type_ids is not None:
        token_type_embeds = model.transformer.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = model.transformer.drop(hidden_states)

    output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    if model.transformer.gradient_checkpointing and model.transformer.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(model.transformer.h, past_key_values)):
            # Model parallel
        if model.transformer.model_parallel:
            torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if 'ffn_gate' in activations.keys():
            gates= block.mlp.act(block.mlp.fc_in(block.ln_1(hidden_states)))
            # gates = decoder_layer.mlp.act_fn(decoder_layer.mlp.gate_proj(hidden_states))
            # hidden_states = decoder_layer.mlp.down_proj(gates * decoder_layer.mlp.up_proj(hidden_states))
            activations['ffn_gate'].append(gates.cpu())
            del gates
            torch.cuda.empty_cache()
        if model.transformer.gradient_checkpointing and model.transformer.training:
            outputs = model.transformer._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                None,
                attention_mask,
                position_ids,
                head_mask[i],
                use_cache,
                output_attentions,
            )
        else:
            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
        if model.transformer.model_parallel:
            for k, v in model.transformer.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != model.transformer.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = model.transformer.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
        # Add last hidden state 
    return activations
