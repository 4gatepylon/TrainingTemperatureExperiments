# %%
###
# NOTE you are going to ahve to install jupyter plugin for VSCode to be able to use cells (every #%% should act like a Jupyer Cell)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import datasets
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
from transformer_lens import evals
import matplotlib.pyplot as plt
import collections
import plotly.graph_objects as go
import os
import contextlib
from typing import Iterable

# %%
torch.set_grad_enabled(False)
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# %%
def in_context_learning_score(model, tokens):
    loss_vec = model(tokens, return_type="loss", loss_per_token=True)
    return (loss_vec[..., 500] - loss_vec[..., 50]).mean()


# %%
# max_checkpoint_idx = 162 # Binary search empirically :P
# checkpoint_indices = list(range(max_checkpoint_idx + 1))
models = ["attn-only-1l", "attn-only-2l", "attn-only-3l"]
# for model_name in models:
#     for checkpoint_idx in checkpoint_indices:
#         model_for_this_checkpoint = HookedTransformer.from_pretrained(model_name, checkpoint_index=checkpoint_idx, device=device)
# %%
if False:
    ###
    # NOTE this is copied from our shit above!
    pile_batch_size = 4
    model = HookedTransformer.from_pretrained(
        "attn-only-2l", device=device
    )  # <--- dummy shit for later
    pile_dataloader = evals.make_pile_data_loader(
        tokenizer=model.tokenizer, batch_size=pile_batch_size
    )
# checkpoint_indices = [10, 25, 35, 60, -1]
# model_to_in_context_learning_scores = {}
# model_to_tokens_trained_on = {}
# for model_name in models:
#     tokens_trained_on = []
#     in_context_learning_scores = []
#     for index in checkpoint_indices:
#         model_for_this_checkpoint = HookedTransformer.from_pretrained(model_name, checkpoint_index=index, device=device)

#         tokens_seen_for_this_checkpoint = model_for_this_checkpoint.cfg.checkpoint_value
#         tokens_trained_on.append(tokens_seen_for_this_checkpoint)


#         in_context_learning_score_for_this_checkpoint = 0
#         # Use subset of dataset for the sake of time
#         num_batches = 2000 // pile_batch_size
#         for i, x in enumerate(pile_dataloader):
#             tokens = x['tokens'].to(device)
#             in_context_learning_score_for_this_checkpoint += in_context_learning_score(model_for_this_checkpoint, tokens).item()
#             if i == num_batches:
#                 break
#         in_context_learning_score_for_this_checkpoint /= num_batches
#         in_context_learning_scores.append(in_context_learning_score_for_this_checkpoint)
#     model_to_in_context_learning_scores[model_name] = in_context_learning_scores
#     model_to_tokens_trained_on[model_name] = tokens_trained_on
# for model_name in model_to_in_context_learning_scores:
#     in_context_learning_scores = model_to_in_context_learning_scores[model_name]
#     tokens_trained_on = model_to_tokens_trained_on[model_name]
#     fig = px.line(x=tokens_trained_on, y=in_context_learning_scores, title=model_name, labels={"x":"Elapsed Training Tokens", "y":"In-Context Learning Scores"}, log_x=True)
#     fig.update_layout(yaxis_range=[-0.6,0.2])
#     fig.add_vrect(x0=3e8, x1=1.5e9, line_width=1, fillcolor="gold", opacity=0.2)
#     fig.show()
# %%
# %%
model1 = HookedTransformer.from_pretrained(models[1], checkpoint_index=0, device=device)
model2 = HookedTransformer.from_pretrained(models[1], checkpoint_index=1, device=device)


# NOTE that each of the parameters should have the proper shape that is (num_heads x in_dim x out_dim,
#      where num_heads would not be there IFF it was going back into resid.)
#
# print([x for x, y in model1.named_parameters()])
# print("W_K Shape", model1.blocks[0].attn.W_K.shape)
# print("b_K Shape", model1.blocks[0].attn.b_K.shape)
# print("W_Q Shape", model1.blocks[0].attn.W_Q.shape)
# print("b_Q Shape", model1.blocks[0].attn.b_Q.shape)
# print("W_O Shape", model1.blocks[0].attn.W_O.shape)
# print("b_O Shape", model1.blocks[0].attn.b_O.shape)
# print("W_V Shape", model1.blocks[0].attn.W_V.shape)
# print("b_V Shape", model1.blocks[0].attn.b_V.shape)
# %%
def mse_between_models(
    model1: nn.Module, model2: nn.Module, param_patts: Optional[List[str] | str] = None
) -> float:
    """
    TODO(Adriano) avoid numerical issues by normalizing iteratively and support per-parameter/component distance calculations.

    Calculate the mean squared error (normalized L2 distance) between two PyTorch models' parameters.

    :param model1: First PyTorch model
    :param model2: Second PyTorch model
    :return: Mean squared error (scalar)
    """
    param_patts = [param_patts] if isinstance(param_patts, str) else param_patts
    squared_diff_sum = 0
    total_params = 0

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            raise ValueError(f"Parameter names do not match: {name1} vs {name2}")
        if param_patts is not None and not any(patt in name1 for patt in param_patts):
            continue  # Ignore this
        diff = param1 - param2
        squared_diff_sum += torch.sum(diff**2).item()
        total_params += param1.numel()

    if total_params == 0:
        raise ValueError("No parameters found in the models")

    mse = squared_diff_sum / total_params
    return mse


def mse_between_models_per_head(
    model1: HookedTransformer, model2: HookedTransformer
) -> np.ndarray:
    """
    Calculate the mean squared error (normalized L2 distance) between two PyTorch models'
    attention head parameters, all combined into one bundle, but split per head. The

    TODO(Adriano) add some better support for the ability to do different subspaces...
        (i.e. so that we can use the previous function instead of re-writing here)

    :return: Numpy Array of mean squared error.
    """
    # Example of what the head patterns should expected to look like:
    # 'blocks.0.attn.W_Q', 'blocks.0.attn.W_O', 'blocks.0.attn.b_Q', 'blocks.0.attn.b_O',
    # 'blocks.0.attn.W_K', 'blocks.0.attn.W_V', 'blocks.0.attn.b_K', 'blocks.0.attn.b_V',
    # 'blocks.1.attn.W_Q', ...
    # TODO(Adriano) we should avoid merging the K, Q, V, O?

    # Same parameter identities and names + same number of blocks, i.e. same strucure
    if not all(
        p1n == p2n and p1.shape == p2.shape
        for (p1n, p1), (p2n, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        )
    ):
        differing_cases = {}
        p1s = {p1n: p1.shape for p1n, p1 in model1.named_parameters()}
        p2s = {p2n: p2.shape for p2n, p2 in model2.named_parameters()}
        for key in p1s:
            if key not in p2s:
                differing_cases[key] = (p1s[key], None)
            elif p1s[key] != p2s[key]:
                differing_cases[key] = (p1s[key], p2s[key])
        for key in p2s:
            if key not in p1s:
                differing_cases[key] = (None, p2s[key])
        raise ValueError(f"Model shapes or names do not match: {differing_cases}")
    num_blocks = len(model1.blocks)
    if num_blocks != len(model2.blocks):
        raise ValueError(
            f"Number of blocks do not match: {num_blocks} vs {len(model2.blocks)}"
        )

    # Shape is correct
    block_patts = [f"blocks.{i}" for i in range(num_blocks)]
    d_model = model1.cfg.d_model
    d_head = model1.cfg.d_head
    n_heads = model1.cfg.n_heads
    assert all(
        (n_heads, d_model, d_head)
        == model1.blocks[i].attn.W_K.shape
        == model1.blocks[i].attn.W_Q.shape
        == model1.blocks[i].attn.W_V.shape
        for i in range(num_blocks)
    )
    assert all(
        (n_heads, d_head, d_model) == model1.blocks[i].attn.W_O.shape
        for i in range(num_blocks)
    )
    assert all(
        (n_heads, d_head)
        == model1.blocks[i].attn.b_K.shape
        == model1.blocks[i].attn.b_Q.shape
        == model1.blocks[i].attn.b_V.shape
        for i in range(num_blocks)
    )
    assert all((d_model,) == model1.blocks[i].attn.b_O.shape for i in range(num_blocks))
    # 1. Initialize Table
    total_mses = -np.ones(
        (num_blocks, n_heads)
    )  # Negative for sanity test later since all MSEs >= 0
    param_names = [
        "W_Q",
        "W_K",
        "W_V",
        "W_O",
        "b_Q",
        "b_K",
        "b_V",
    ]  # , "b_O"] # Ignore b_O right now because it's not really per-head

    # 3. Fill the table
    for block_idx, block_patt in enumerate(block_patts):
        params1 = [
            model1.blocks[block_idx].attn.__getattr__(pname) for pname in param_names
        ]
        params2 = [
            model2.blocks[block_idx].attn.__getattr__(pname) for pname in param_names
        ]
        assert all(
            p.shape[0] == n_heads and len(p.shape) >= 2 for p in params1 + params2
        )
        for head_idx in range(n_heads):
            # Get params
            head_params1 = [p[head_idx] for p in params1]
            head_params2 = [p[head_idx] for p in params2]
            # Get averaging constant
            head_numel1 = sum(p.numel() for p in head_params1)
            head_numel2 = sum(p.numel() for p in head_params2)
            assert head_numel1 == head_numel2
            # Get the differences
            head_diffs: List[torch.Tensor] = [
                p1 - p2 for p1, p2 in zip(head_params1, head_params2)
            ]
            head_diff2: float = sum(torch.sum(diff**2).item() for diff in head_diffs)
            # Calculate MSE and upate table
            head_mse = head_diff2 / head_numel1
            total_mses[block_idx, head_idx] = head_mse

    # 4. Sanity and Return
    assert total_mses.min() >= 0
    return total_mses


###
#  TEST
print("Global MSE", mse_between_models(model1, model2))
print("Per-Head MSE", mse_between_models_per_head(model1, model2))


# %%
def mse_between_model_checkpoints_per_head_over_time(  # time = training
    model_name: str,
    per_head: bool = False,
    indices: Iterable[int] = list(range(163)),
    break_on_index_error: bool = True,
    must_hit_max_index: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a timeseries of the MSE of the weight changes (i.e. between two
    training steps, calculate the difference in the weights and take the squared value,
    then divide by number of parameters: not really MSE, more like norm, divided by
    parameter and not yet mean-ed).

    TODO(Adriano) add some better support for L1, etc...
    TODO(Adriano) for the very first step the delta in tokens trained might be a
        little odd?
    TODO(Adriano) add some normalization modes?

    Args:
        model1_name (str): Name of the model to get from transformer_lens.
        per_head (bool, optional): Whether the first tensor should be
            (n_steps, block_idx, head_idx) or just be (n_steps,); in the former
            case it represents the per-head MSEs, while in the latter it is the
            global weight MSE. Defaults to False.
        indices (Iterable[int], optional): a list of CHECKPOINT indices that transformer
            lens expects as we look for diff.
            checkpoints. Defaults to list(range(163)).
        break_on_index_error (bool, optional): Whether or not it is OK to overflow
            the indices; if so will just break; otherwise, throw. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: one number per edge in each output ndarray
            (edge is per pair of cons. indices). The first array is the actual norms
            (n_heads x ...) where ... is either nothing for global, or some sort of table
            for per-component/per-head. The second array is just a list of the number
            of tokens that are claimed to have been trained up until
            at this point. This helps us understand how to interpret a change better.
    """
    indices = list(indices)
    prev_model = prev_model = HookedTransformer.from_pretrained(
        model_name, checkpoint_index=indices[0], device=device
    )
    differences = []
    tokens_trained_on = []
    hit_max_index = None

    for i in tqdm.tqdm(indices[1:]):
        with open(os.devnull, "w") as devnull:
            # Don't logspam
            with contextlib.redirect_stdout(devnull):
                try:
                    # 0. Get the new model
                    new_model = HookedTransformer.from_pretrained(
                        model_name, checkpoint_index=i, device=device
                    )
                    # 1. Update tokens rained on
                    tokens_seen_for_this_checkpoint = new_model.cfg.checkpoint_value
                    tokens_trained_on.append(tokens_seen_for_this_checkpoint)
                    # 2. Update differences
                    diff = (
                        mse_between_models(prev_model, new_model)
                        if not per_head
                        else mse_between_models_per_head(prev_model, new_model)
                    )
                    assert isinstance(diff, float) or isinstance(diff, np.ndarray)
                    if isinstance(diff, float):
                        diff = np.array([diff])
                    differences.append(diff)
                    # 3. Update prev_model
                    prev_model = new_model
                except IndexError as e:
                    hit_max_index = i
                    if break_on_index_error:
                        break
                    raise e
    assert not must_hit_max_index or hit_max_index is not None
    x, y = np.stack(differences).squeeze(0), np.array(tokens_trained_on)
    assert x.shape[0] == y.shape[0]
    return x, y


def mse_between_model_checkpoints_per_head_over_time_per_models(
    model_names: List[str], **kwargs
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Do the previous method but for multiple models.

    TODO(Adriano) add support for checking that they all are on the same exact
    training steps (shouldn't matter but ok whatever).
    """
    return {
        model_name: mse_between_model_checkpoints_per_head_over_time(
            model_name, **kwargs
        )
        for model_name in model_names
    }


# %%
index_iterator = list(range(1, 200, 8))  # <--- takes up HELLA space
# index_iterator = [160, 161, 162, 163] # Debug
# index_iterator = list(range(8)) + [10, 25, 30, 35, 40, 45, 50, 55, 60, -1]
# NOTE: per small model, downloading the model MIGHT take up to around 1GB/checkpoint,
# which means that if you want around 200 for 3 models -> 600 checkpoints, you need around
# 1TB free
print("*"*100)
print("GLOBAL")
print("*"*100)
model2diffs_tto_global = mse_between_model_checkpoints_per_head_over_time_per_models(
    models,
    per_head=False,
    indices=index_iterator,
    break_on_index_error=True,
    must_hit_max_index=True,
)
# %%
print("*"*100)
print("PER HEAD")
print("*"*100)
model2diffs_tto_per_head = mse_between_model_checkpoints_per_head_over_time_per_models(
    models,
    per_head=True,
    indices=index_iterator,
    break_on_index_error=True,
    must_hit_max_index=True,
)
# %%
def plot_mse_over_time(model2diffs_tto: dict[str, tuple[np.ndarray, np.ndarray]], log_x: bool = False):
    for model_name, (diffs, tto) in model2diffs_tto.items():
        assert len(diffs) == len(tto)
        fig = go.Figure(layout={'title':model_name})
        fig.update_xaxes(title=f"Elapsed Training Tokens (log={log_x})")
        fig.update_yaxes(title="Prefix Matching Score")
        # TODO(Adriano) add better support for flexibility here?
        # fig.update_layout(yaxis_range=[0.0, 1.0])
        fig.add_vrect(x0=3e8, x1=1.5e9, line_width=1, fillcolor="gold", opacity=0.2)
        if len(diffs.shape) == 1:
            fig.add_trace(go.Scatter(x=tto, y=diffs, name="global"))
        else:
            diffs = einops.rearrange(diffs, "n_steps ... -> ... n_steps")
            shape_except_last = diffs.shape[:-1]
            it = np.nditer(np.zeros(shape_except_last), flags=['multi_index'])
            while not it.finished:
                # Access the sub-array for the current indices
                sub_array = diffs[it.multi_index]
                assert sub_array.ndim == 1 and sub_array.shape[0] == len(tto)
                assert isinstance(sub_array, np.ndarray)
                assert isinstance(it.multi_index, tuple)
                fig.add_trace(go.Scatter(x=tto, y=sub_array, name=f"component {it.multi_index}"))
                
                it.iternext()
        fig.show()
# %%
plot_mse_over_time(model2diffs_tto_global, log_x=False)
# %%
# list(model2diffs_tto_per_head.values())[0][0].shape
# model2diffs_tto_per_head = {
#     x : (np.expand_dims(y, axis=1), z) for x, (y, z) in model2diffs_tto_per_head.items()
# }
plot_mse_over_time(model2diffs_tto_per_head, log_x=False)

# %%
