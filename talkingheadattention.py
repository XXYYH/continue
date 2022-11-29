#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the full attention similar to the one implemented by PyTorch's
MultiHeadAttention module. Note that this module is to be used in conjuction
with the `fast_transformers.attention.attention_layer.AttentionLayer` in order
to work."""

from math import sqrt

import torch
from torch.nn import Dropout, Module, Conv2d

from fast_transformers.attention_registry import AttentionRegistry, RecurrentAttentionRegistry, Optional, Float, \
    EventDispatcherInstance

from fast_transformers.events import EventDispatcher, AttentionEvent
from fast_transformers.recurrent._utils import check_state 


class TalkingHeadAttention(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, heads=8, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=""):
        super(TalkingHeadAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        self.pre_softmax_talking_heads  = Conv2d(heads, heads, 1, bias = False)
        self.post_softmax_talking_heads = Conv2d(heads, heads, 1, bias = False)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Scale the queries instead of applying the softmax temperature to the
        # dot products
        queries = queries * softmax_temp

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        QK = self.pre_softmax_talking_heads(QK)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        if not key_lengths.all_ones:
            QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(QK, dim=-1))
        A = self.post_softmax_talking_heads(A)
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "talking", TalkingHeadAttention,
    [
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)

class RecurrentTalkingHeadAttention(Module):
    """Implement the full softmax attention as a recurrent module.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, heads=8, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=""):
        super(RecurrentTalkingHeadAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        self.pre_softmax_talking_heads  = Conv2d(heads, heads, 1, bias = False)
        self.post_softmax_talking_heads = Conv2d(heads, heads, 1, bias = False)

    def forward(self, query, key, value, state=None, memory=None):
        # Normalize state/memory
        state = check_state(state, memory)

        # Extract some shapes and compute the temperature
        N, H, E = query.shape
        _, _, D = value.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Aggregate the list of keys and values
        if state is not None:
            keys, values = state
            keys = torch.cat([keys, key[:, :, None]], dim=2)
            values = torch.cat([values, value[:, :, None]], dim=2)
        else:
            keys = key[:, :, None]
            values = value[:, :, None]

        _, _, S, _ = keys.shape
        # Compute the unnormalized attention
        QK = torch.einsum("nhe,nhse->nhs", query, keys)
        # QK=QK.permute(1,0,2)
        # QK = QK.view(N, H, S, -1)
        QK = QK.view(N, H, 1, S)
        QK = self.pre_softmax_talking_heads(QK)
        # QK=QK.permute(1,0,2)
        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        # A=A.permute(1,0,2)
        A = self.post_softmax_talking_heads(A)
        # A=A.permute(1,0,2)
        A = A.view(N, H, S)
        V = torch.einsum("nhs,nhsd->nhd", A, values).contiguous()

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V, [keys, values]


# Register the attention implementation so that it becomes available in our
# builders
RecurrentAttentionRegistry.register(
    "talking", RecurrentTalkingHeadAttention,
    [
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)