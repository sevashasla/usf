import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt, 
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 # semantic
                 num_semantic_classes=-1,
                 num_layers_semantic=2,
                 hidden_dim_semantic=64,
                 # uncertainty
                 beta_min=0.01,
                 bound=1,
                 Ngen=10, 
                 arc_rgb=0, 
                 arc_smntc=0,
                 **kwargs,
                 ):
        '''
        architecture_* - how deep to take features from the last layer
        
        arc_rgb:
            [0, 1]: hidden_dim_color
            2: in_dim_dir + geo_feat_dim
            3: geo_feat_dim
        
        arc_smntc:
            0: hidden_dim_semantic
            1: geo_feat_dim
        '''
        super().__init__(bound, **kwargs)

        self.use_uncert = opt.use_uncert
        self.use_semantic = opt.use_semantic
        self.use_semantic_uncert = opt.use_semantic_uncert
        self.first_encoding = encoding

        self.num_semantic_classes = num_semantic_classes
        self.Ngen = Ngen

        self.arc_rgb = arc_rgb
        self.arc_smntc = arc_smntc

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

        if self.use_uncert:
            self.in_uncert_dim = [self.hidden_dim_color, self.hidden_dim_color, self.in_dim_dir + self.geo_feat_dim, self.geo_feat_dim][arc_rgb]
            self.beta_min = beta_min
            self.layer_uncertainty = nn.Linear(self.in_uncert_dim, 1)

        # semantic network
        if self.use_semantic:
            self.num_layers_semantic = num_layers_semantic
            self.hidden_dim_semantic = hidden_dim_semantic
            semantic_net = []
            for l in range(num_layers_semantic):
                if l == 0:
                    in_dim = self.geo_feat_dim
                else:
                    in_dim = self.hidden_dim_semantic

                if l == num_layers_semantic - 1:
                    out_dim = self.num_semantic_classes
                else:
                    out_dim = self.hidden_dim_semantic
                semantic_net.append(nn.Linear(in_dim, out_dim, bias=False))
            self.semantic_net = nn.ModuleList(semantic_net)
        
        # uncertainty semantic
        if self.use_semantic_uncert:
            self.in_smntc_uncert_dim = [self.hidden_dim_semantic, self.geo_feat_dim][arc_smntc]
            self.beta_min = beta_min
            self.layer_semantic_uncertainty = nn.Linear(self.hidden_dim_semantic, 1)


    def semantic_postprocess_prob(self, mu, sigma=None):
        '''
        2 options:
            1) deterministic if not use_semantic_uncert
            2) we suppose that logits ~ Normal(mu, sigma^2)

        ---
        Arguments

        mu: [B, ..., SC]
            - mean of logits

        sigma: [B, ...]
            - sigma of logits
        '''

        if self.use_semantic_uncert:
            init_shape = mu.shape
            mu = mu.view(init_shape[0], -1, init_shape[-1])
            sigma = sigma.view(init_shape[0], -1, 1)
            B, N, SC = mu.shape
            probs = torch.zeros_like(mu)
            # cycle to avoid OOM
            for _ in range(self.Ngen):
                epsilons = torch.randn(B, N, SC, dtype=mu.dtype, device=mu.device)
                logits = mu + sigma * epsilons
                probs = probs + F.softmax(logits, dim=-1)
            probs = (probs / self.Ngen).view(init_shape)
        else:
            probs = F.softmax(mu, dim=-1)
        return probs
        

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        sigma_pred = self.density(x)        
        sigma = sigma_pred["sigma"]
        geo_feat = sigma_pred["geo_feat"]

        # color
        color_pred = self.color(x, d, get_feat=geo_feat)
        if self.use_semantic:
            semantic_pred = self.semantic_pred(x, geo_feat=geo_feat)
        else:
            semantic_pred = {
                "smntc": None,
                "smntc_uncert": None,
            }

        return {
                "sigma": sigma, 
                **color_pred,
                **semantic_pred,
            }

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        if self.first_encoding == "hashgrid":
            x = self.encoder(x, bound=self.bound)
        else:
            x = self.encoder(x)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            
            if self.use_uncert:
                uncert = torch.zeros(mask.shape[0], 1, dtype=x.dtype, device=x.device)
            else:
                uncert = None
            
            # in case of empty mask
            if not mask.any():
                return {
                    "color": rgbs,
                    "uncert": uncert,
                }
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            if self.use_uncert and l + self.arc_rgb == self.num_layers_color - 1:
                h_uncert = self.layer_uncertainty(h)
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
            else:
                h_color = h
        if self.use_semantic and self.arc_rgb >= self.num_layers_color:
            h_uncert = self.layer_uncertainty(geo_feat)
        
        # sigmoid activation for rgb
        h_color = torch.sigmoid(h_color)
        if self.use_uncert:
            h_uncert = F.softplus(h_uncert) + self.beta_min
        else:
            h_uncert = None

        if mask is not None:
            rgbs[mask] = h_color.to(rgbs.dtype) # fp16 --> fp32
            if self.use_uncert:
                uncert[mask] = h_uncert.to(uncert.dtype) # fp16 --> fp32
            else:
                uncert = None
        else:
            rgbs = h_color
            uncert = h_uncert

        return {
            "color": rgbs,
            "uncert": uncert,
        }

    def semantic_pred(self, x, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if not self.use_semantic:
            raise RuntimeError("call of semantic pred, but use_semantic is false!")

        if mask is not None:
            smntc = torch.zeros(mask.shape[0], self.num_semantic_classes, dtype=x.dtype, device=x.device) # [N, SC]
            if self.use_semantic_uncert:
                smntc_uncert = torch.zeros(mask.shape[0], 1, dtype=x.dtype, device=x.device) # [N, SC]
            else:
                smntc_uncert = None
            # in case of empty mask
            if not mask.any():
                return {
                    "smntc": smntc,
                    "smntc_uncert": smntc_uncert,
                }
            
            x = x[mask]
            geo_feat = geo_feat[mask]

        h = geo_feat
        for l in range(self.num_layers_semantic):
            if self.use_semantic_uncert and l + self.arc_smntc == self.num_layers_semantic - 1:
                h_smntc_uncert = self.layer_semantic_uncertainty(h)
            h = self.semantic_net[l](h)
            if l != self.num_layers_semantic - 1:
                h = F.relu(h, inplace=True)
            else:
                h_smntc = h

        if mask is not None:
            smntc[mask] = h_smntc.to(smntc.dtype) # fp16 --> fp32
            if self.use_semantic_uncert:
                smntc_uncert[mask] = h_smntc_uncert.to(smntc_uncert.dtype) # fp16 --> fp32
            else:
                smntc_uncert = None
        else:
            smntc = h_smntc
            smntc_uncert = h_smntc_uncert

        return {
                "smntc": smntc,
                "smntc_uncert": smntc_uncert,
            }

    # optimizer utils
    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.use_uncert:
            params.append({'params': self.layer_uncertainty.parameters(), 'lr': lr})
        if self.use_semantic:
            params.append({'params': self.semantic_net.parameters(), 'lr': lr})
        if self.use_semantic_uncert:
            params.append({'params': self.layer_semantic_uncertainty.parameters(), 'lr': lr})
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
