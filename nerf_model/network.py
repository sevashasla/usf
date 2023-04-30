import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
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
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        self.use_semantic = num_semantic_classes > 0
        self.num_semantic_classes = num_semantic_classes

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
        
        # uncertainty color
        self.beta_min = beta_min
        self.layer_uncertainty = nn.Linear(self.geo_feat_dim, 1)

        # uncertainty semantic
        self.beta_min = beta_min
        self.layer_semantic_uncertainty = nn.Linear(self.geo_feat_dim, 1)
        

    @staticmethod
    def semantic_postprocess_prob(mu, sigma, Ngen=1000):
        '''
        we suppose that logits ~ Normal(mu, sigma^2)

        ---
        Arguments

        mu: [B, ???, n]
            - mean of logits

        sigma: [B, ???, n]
            - sigma of logits
        
        Ngen: int
            - how many vectors to generate
        '''

        init_shape = mu.shape
        mu = mu.view(init_shape[0], -1, init_shape[-1])
        sigma = sigma.view(init_shape[0], -1, init_shape[-1])
        B, N, n = mu.shape
        epsilons = torch.randn(Ngen, B, N, n, dtype=mu.dtype, device=mu.device)
        logits = mu + sigma * epsilons
        probs = F.softmax(logits, dim=-1).mean(dim=0).view(init_shape)
        return probs
        

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
            else:
                prev_last = h

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]


        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        # semantic
        if self.use_semantic:
            h = geo_feat
            for l in range(self.num_layers_semantic):
                h = self.semantic_net[l](h)
                if l != self.num_layers_semantic - 1:
                    h = F.relu(h, inplace=True)
            semantic = h
        else:
            semantic = None

        # uncertainty
        uncertainty = self.layer_uncertainty(geo_feat)
        uncertainty = F.softplus(uncertainty) + self.beta_min
        
        # semantic uncertainty
        semantic_uncertainty = self.layer_semantic_uncertainty(geo_feat)
        semantic_uncertainty = F.softplus(semantic_uncertainty) + self.beta_min

        return sigma, color, semantic, uncertainty, semantic_uncertainty

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
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
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    def semantic_pred(self, x, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if not self.use_semantic:
            return None

        if mask is not None:
            smntc = torch.zeros(mask.shape[0], self.num_semantic_classes, dtype=x.dtype, device=x.device) # [N, SC]
            # in case of empty mask
            if not mask.any():
                return smntc
            x = x[mask]
            geo_feat = geo_feat[mask]

        h = geo_feat
        for l in range(self.num_layers_semantic):
            h = self.semantic_net[l](h)
            if l != self.num_layers_semantic - 1:
                h = F.relu(h, inplace=True)

        if mask is not None:
            smntc[mask] = h.to(smntc.dtype) # fp16 --> fp32
        else:
            smntc = h

        return smntc

    def uncertainty_pred(self, x, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            uncert = torch.zeros(mask.shape[0], 1, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return uncert
            x = x[mask]
            geo_feat = geo_feat[mask]
        
        # uncertainty
        h = geo_feat
        h = self.layer_uncertainty(h)
        h = F.softplus(h) + self.beta_min

        if mask is not None:
            uncert[mask] = h.to(uncert.dtype) # fp16 --> fp32
        else:
            uncert = h

        return uncert
    
    def semantic_uncertainty_pred(self, x, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            semantic_uncert = torch.zeros(mask.shape[0], 1, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return semantic_uncert
            x = x[mask]
            geo_feat = geo_feat[mask]
        
        # uncertainty
        h = geo_feat
        h = self.layer_semantic_uncertainty(h)
        h = F.softplus(h) + self.beta_min

        if mask is not None:
            semantic_uncert[mask] = h.to(semantic_uncert.dtype) # fp16 --> fp32
        else:
            semantic_uncert = h

        return semantic_uncert


    # optimizer utils
    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.layer_uncertainty.parameters(), 'lr': lr}, 
        ]
        if self.use_semantic:
            params.append({'params': self.semantic_net.parameters(), 'lr': lr})
            params.append({'params': self.layer_semantic_uncertainty.parameters(), 'lr': lr})
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
