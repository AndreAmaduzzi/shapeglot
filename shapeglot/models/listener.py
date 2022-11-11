"""
The MIT License (MIT)
Originally created sometime in 2019.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
from torch import nn


class Listener(nn.Module):
    def __init__(self, language_encoder, image_encoder, mlp_decoder, pc_encoder=None):
        super(Listener, self).__init__()
        self.language_encoder = language_encoder
        self.image_encoder = image_encoder
        self.pc_encoder = pc_encoder
        self.logit_encoder = mlp_decoder

    def forward(self, item_ids, padded_tokens, dropout_rate=0.5):
        visual_feats = self.image_encoder(item_ids, dropout_rate)
        lang_feats = self.language_encoder(padded_tokens, init_feats=visual_feats)  # lang_feats: list with 3 tensors of size [B, 100]

        if self.pc_encoder is not None:
            pc_feats = self.pc_encoder(item_ids, dropout_rate, pre_drop=False)      # pc_feats size [B, 3, 100]
        else:
            pc_feats = None

        logits = []
        for i, l_feats in enumerate(lang_feats):
            if pc_feats is not None:
                feats = torch.cat([l_feats, pc_feats[:, i]], 1) # feats size: [B, 200]
            else:
                feats = l_feats

            logits.append(self.logit_encoder(feats))    # logit encoder: MLP mapping from 200 to 100, then 50, then 1
        return torch.cat(logits, 1)                     # logits: list with 3 tensors of shape [B, 1]

class T2S_Listener(nn.Module):
    def __init__(self, language_encoder, pc_encoder, mlp_decoder):
        super(T2S_Listener, self).__init__()
        self.language_encoder = language_encoder
        self.pc_encoder = pc_encoder
        self.logit_encoder = mlp_decoder

    def forward(self, item_ids, padded_tokens, dropout_rate=0.5):
        lang_feats = self.language_encoder(padded_tokens, init_feats=visual_feats)
        pc_feats = self.pc_encoder(item_ids, dropout_rate, pre_drop=False)


        logits = []
        for i, l_feats in enumerate(lang_feats):
            feats = torch.cat([l_feats, pc_feats[:, i]], 1)
            logits.append(self.logit_encoder(feats))
        return torch.cat(logits, 1)
        
