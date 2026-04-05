from abc import abstractmethod

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import checkpoint, conv_nd, linear, avg_pool_nd, zero_module, normalization, timestep_embedding
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to children that
    support it.  Also passes cond_feat to CrossModalAttentionBlock layers.
    All other layers (ResBlock, AttentionBlock, Upsample, Downsample, etc.)
    are unchanged — they receive only (x, emb) or (x,) as before.
    """

    def forward(self, x, emb, cond_feat=None):
        for layer in self:
            if isinstance(layer, CrossModalAttentionBlock):
                x = layer(x, cond_feat=cond_feat)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True, use_freq=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d
        self.use_freq = use_freq
        self.idwt = IDWT_3D("haar")
        if use_conv:
            self.conv = conv_nd(dims, self.channels * 7, self.out_channels * 7, 3, padding=1, groups=7)

    def forward(self, x):
        if isinstance(x, tuple):
            skip = x[1]
            x = x[0]
        else:
            skip = None
        assert x.shape[1] == self.channels
        if self.use_freq and skip is not None:
            if self.use_conv:
                skip = self.conv(th.cat(skip, dim=1) / 3.) * 3.
                skip = tuple(th.chunk(skip, 7, dim=1))
            x = self.idwt(3. * x, skip[0], skip[1], skip[2], skip[3], skip[4], skip[5], skip[6])
        else:
            if self.dims == 3 and self.resample_2d:
                x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x, None


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True, use_freq=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.use_freq = use_freq
        self.dwt = DWT_3D("haar")
        stride = (1, 2, 2) if dims == 3 and resample_2d else 2
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        elif self.use_freq:
            self.op = self.dwt
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        if self.use_freq:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.op(x)
            x = (LLL / 3., (LLH, LHL, LHH, HLL, HLH, HHL, HHH))
        else:
            x = self.op(x)
        return x


class WaveletDownsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = conv_nd(3, self.in_ch * 8, self.out_ch, 3, stride=1, padding=1)
        self.dwt = DWT_3D('haar')

    def forward(self, x):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x)
        x = th.cat((LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH), dim=1) / 3.
        return self.conv(x)


# ---------------------------------------------------------------------------
# Parallel condition encoder — the key to real cross-modal attention
# ---------------------------------------------------------------------------

class CondEncoder(nn.Module):
    """
    Lightweight parallel encoder for the condition wavelet subbands only.

    Input:  24ch = 3 condition modalities x 8 wavelet subbands (the 3:1
            ratio from cWDM's input design, kept pure — no target channels).
    Output: one feature tensor per UNet level, at the same spatial resolution
            as the corresponding decoder level, used as K/V in
            CrossModalAttentionBlock.

    By keeping target and condition in separate encoder paths, the
    cross-modal attention can genuinely ask "what in the condition modalities
    is relevant to reconstructing the target?" rather than attending over
    an arbitrarily sliced mixed feature map.
    """

    def __init__(self, cond_channels=24, channel_mult=(1, 2, 4, 4),
                 model_channels=64, dims=3, num_groups=32):
        super().__init__()
        self.channel_mult = channel_mult
        self.model_channels = model_channels

        # Project 24ch condition input to model_channels
        self.input_proj = conv_nd(dims, cond_channels, model_channels, 3, padding=1)

        # One downsample block per UNet level — mirrors main UNet spatial schedule
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for mult in channel_mult:
            out_ch = mult * model_channels
            g_in  = min(num_groups, ch);     [g_in  := g_in  // 2 for _ in range(100) if ch      % g_in  != 0]
            g_out = min(num_groups, out_ch); [g_out := g_out // 2 for _ in range(100) if out_ch % g_out != 0]
            block = nn.Sequential(
                normalization(ch,     g_in),
                nn.SiLU(),
                conv_nd(dims, ch,     out_ch, 3, padding=1),
                normalization(out_ch, g_out),
                nn.SiLU(),
                conv_nd(dims, out_ch, out_ch, 3, stride=2, padding=1),
            )
            self.down_blocks.append(block)
            ch = out_ch

    def forward(self, cond):
        """
        Returns feats: list of tensors [finest, ..., coarsest] — one per level.
        """
        feats = []
        h = self.input_proj(cond)
        for block in self.down_blocks:
            h = block(h)
            feats.append(h)
        return feats  # feats[0]=after level-0 (finest), feats[-1]=coarsest


# ---------------------------------------------------------------------------
# Standard UNet building blocks (unchanged from original)
# ---------------------------------------------------------------------------

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=True,
                 use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False,
                 num_groups=32, resample_2d=True, use_freq=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint
        self.up = up
        self.down = down
        self.num_groups = num_groups
        self.use_freq = use_freq

        self.in_layers = nn.Sequential(
            normalization(channels, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims, resample_2d=resample_2d, use_freq=self.use_freq)
            self.x_upd = Upsample(channels, False, dims, resample_2d=resample_2d, use_freq=self.use_freq)
        elif down:
            self.h_upd = Downsample(channels, False, dims, resample_2d=resample_2d, use_freq=self.use_freq)
            self.x_upd = Downsample(channels, False, dims, resample_2d=resample_2d, use_freq=self.use_freq)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, self.num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, temb):
        if isinstance(x, tuple):
            hSkip = x[1]
        else:
            hSkip = None

        if self.updown:
            if self.up:
                x = x[0]
            h = self.in_layers(x)
            if self.up:
                h = (h, hSkip)
                x = (x, hSkip)
            h, hSkip = self.h_upd(h)
            x, xSkip = self.x_upd(x)
        else:
            if isinstance(x, tuple):
                x = x[0]
            h = self.in_layers(x)

        emb_out = self.emb_layers(temb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        out = self.skip_connection(x) + h
        return out, hSkip


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False,
                 use_new_attention_order=False, num_groups=32):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        received_tuple = False
        skip = None
        if isinstance(x, tuple):
            received_tuple = True
            x, skip = x
        out = checkpoint(self._forward, (x,), self.parameters(), True)
        if received_tuple:
            return out, skip
        return out

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# ---------------------------------------------------------------------------
# True cross-modal attention block
# ---------------------------------------------------------------------------

class CrossModalAttentionBlock(nn.Module):
    """
    Cross-modal attention where:
      Q  = main UNet decoder features (target reconstruction path)
      K,V = CondEncoder features at the same spatial scale (condition-only path)

    The 3:1 ratio from cWDM is preserved because:
      - Main UNet path started from 32ch (8 target + 24 condition concatenated)
      - CondEncoder path started from exactly the 24ch condition slice
      - The attention bridges them: target features query the condition features
        at each spatial scale, just as cWDM's input design intends but now in
        feature space rather than raw channel concatenation.

    The residual update is applied only to the main UNet features (x).
    Condition features are read-only — they guide but are not modified.
    """

    def __init__(self, channels, cond_channels, num_heads=1, num_groups=32, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.cond_channels = cond_channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        assert channels % num_heads == 0, \
            f"CrossModalAttentionBlock: channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.head_dim = channels // num_heads

        # GroupNorm for query and key/value separately
        q_groups  = min(num_groups, channels);      
        while channels % q_groups != 0:      q_groups  //= 2
        kv_groups = min(num_groups, cond_channels); 
        while cond_channels % kv_groups != 0: kv_groups //= 2

        self.norm_q  = normalization(channels,      q_groups)
        self.norm_kv = normalization(cond_channels, kv_groups)

        # Q from main UNet features, K/V from condition encoder features
        self.q_proj   = conv_nd(1, channels,      channels,     1)
        self.kv_proj  = conv_nd(1, cond_channels, channels * 2, 1)
        # Zero-init so the block starts as identity and learns to activate
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_feat=None):
        received_tuple = False
        skip = None
        if isinstance(x, tuple):
            received_tuple = True
            x, skip = x

        if cond_feat is None:
            # Safety fallback: no condition available, pass through unchanged
            if received_tuple:
                return x, skip
            return x

        out = checkpoint(self._forward, (x, cond_feat), self.parameters(), self.use_checkpoint)
        if received_tuple:
            return out, skip
        return out

    def _forward(self, x, cond_feat):
        b, c, *spatial = x.shape
        N = int(np.prod(spatial))

        x_flat    = x.reshape(b, c, -1)
        cond_flat = cond_feat.reshape(b, self.cond_channels, -1)

        assert x_flat.shape[2] == cond_flat.shape[2], (
            f"CrossModalAttentionBlock spatial mismatch: "
            f"UNet {x_flat.shape[2]} vs CondEncoder {cond_flat.shape[2]}"
        )

        q  = self.q_proj(self.norm_q(x_flat))        # [B, C, N]
        kv = self.kv_proj(self.norm_kv(cond_flat))   # [B, 2C, N]
        k, v = kv.chunk(2, dim=1)                    # each [B, C, N]

        scale = self.head_dim ** -0.5
        q = q.reshape(b * self.num_heads, self.head_dim, N)
        k = k.reshape(b * self.num_heads, self.head_dim, N)
        v = v.reshape(b * self.num_heads, self.head_dim, N)

        attn   = th.einsum('bdn,bdm->bnm', q, k) * scale
        attn   = th.softmax(attn.float(), dim=-1).type(q.dtype)
        h_attn = th.einsum('bnm,bdm->bdn', attn, v)
        h_attn = h_attn.reshape(b, c, N)

        out = x_flat + self.proj_out(h_attn)
        return out.reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# Main WavUNet model
# ---------------------------------------------------------------------------

class WavUNetModel(nn.Module):
    """
    Full wavelet-domain UNet for conditional MRI synthesis.

    When use_cross_attn=True:
    - A parallel CondEncoder encodes x[:, 8:, ...] (the 24ch condition
      subbands = 3 modalities x 8 wavelet bands) independently.
    - In the decoder, CrossModalAttentionBlock uses Q from the main UNet
      path and K/V from CondEncoder — true cross-modal attention that
      preserves the 3:1 target/condition ratio from cWDM's input design.

    The main UNet still receives the full 32ch input (cWDM concatenation)
    so the proven cWDM conditioning mechanism is fully intact.
    """

    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True,
                 dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=1,
                 num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False,
                 resblock_updown=False, use_new_attention_order=False, num_groups=32,
                 bottleneck_attention=True, resample_2d=True, additive_skips=False,
                 decoder_device_thresh=0, use_freq=False, progressive_input='residual',
                 use_cross_attn=False, cond_channels=24):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_groups = num_groups
        self.bottleneck_attention = bottleneck_attention
        self.devices = None
        self.decoder_device_thresh = decoder_device_thresh
        self.additive_skips = additive_skips
        self.use_freq = use_freq
        self.progressive_input = progressive_input
        self.use_cross_attn = use_cross_attn
        self.cond_channels = cond_channels  # 24 = 3 modalities x 8 wavelet subbands

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim))

        # Parallel condition encoder — keeps condition features separate from
        # target features, enabling true cross-modal attention in the decoder
        if self.use_cross_attn:
            self.cond_encoder = CondEncoder(
                cond_channels=cond_channels,       # 24ch = 3 x 8 subbands
                channel_mult=channel_mult,
                model_channels=model_channels,
                dims=dims,
                num_groups=num_groups,
            )

        # Input block
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        input_pyramid_channels = in_channels
        ds = 1

        # Encoder
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(channels=ch, emb_channels=time_embed_dim, dropout=dropout,
                             out_channels=mult * model_channels, dims=dims,
                             use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                             num_groups=self.num_groups, resample_2d=resample_2d, use_freq=self.use_freq)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                                                 num_head_channels=num_head_channels,
                                                 use_new_attention_order=use_new_attention_order,
                                                 num_groups=self.num_groups))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            out_ch = ch
            layers = [
                ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                         use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                         down=True, num_groups=self.num_groups, resample_2d=resample_2d, use_freq=self.use_freq)
                if resblock_updown else
                Downsample(ch, conv_resample, dims=dims, out_channels=out_ch,
                           resample_2d=resample_2d, use_freq=self.use_freq)
            ]
            self.input_blocks.append(TimestepEmbedSequential(*layers))

            layers = []
            if self.progressive_input == 'residual':
                layers.append(WaveletDownsample(in_ch=input_pyramid_channels, out_ch=out_ch))
                input_pyramid_channels = out_ch
            self.input_blocks.append(TimestepEmbedSequential(*layers))

            ch = out_ch
            input_block_chans.append(ch)
            ds *= 2
            self._feature_size += ch

        self.input_block_chans_bk = input_block_chans[:]

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm, num_groups=self.num_groups,
                     resample_2d=resample_2d, use_freq=self.use_freq),
            *([AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                              num_head_channels=num_head_channels,
                              use_new_attention_order=use_new_attention_order,
                              num_groups=self.num_groups)] if self.bottleneck_attention else []),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm, num_groups=self.num_groups,
                     resample_2d=resample_2d, use_freq=self.use_freq),
        )
        self._feature_size += ch

        # Decoder
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # CondEncoder output at this spatial scale has mult*model_channels channels
            cond_ch_at_level = mult * model_channels

            for i in range(num_res_blocks + 1):
                if i != num_res_blocks:
                    mid_ch = model_channels * mult
                    layers = [
                        ResBlock(ch, time_embed_dim, dropout, out_channels=mid_ch, dims=dims,
                                 use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                                 num_groups=self.num_groups, resample_2d=resample_2d, use_freq=self.use_freq)
                    ]
                    if ds in attention_resolutions:
                        layers.append(
                            AttentionBlock(mid_ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample,
                                           num_head_channels=num_head_channels,
                                           use_new_attention_order=use_new_attention_order,
                                           num_groups=self.num_groups)
                        )
                        if self.use_cross_attn:
                            layers.append(
                                CrossModalAttentionBlock(
                                    channels=mid_ch,
                                    cond_channels=cond_ch_at_level,
                                    num_heads=num_heads_upsample,
                                    num_groups=self.num_groups,
                                    use_checkpoint=use_checkpoint,
                                )
                            )
                    ch = mid_ch
                else:
                    out_ch = ch
                    layers.append(
                        ResBlock(mid_ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                                 up=True, num_groups=self.num_groups, resample_2d=resample_2d, use_freq=self.use_freq)
                        if resblock_updown else
                        Upsample(mid_ch, conv_resample, dims=dims, out_channels=out_ch,
                                 resample_2d=resample_2d, use_freq=self.use_freq)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                mid_ch = ch

        # Output residual blocks
        self.out_res = nn.ModuleList([])
        for i in range(num_res_blocks):
            self.out_res.append(TimestepEmbedSequential(
                ResBlock(ch, time_embed_dim, dropout, out_channels=ch, dims=dims,
                         use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                         num_groups=self.num_groups, resample_2d=resample_2d, use_freq=self.use_freq)
            ))

        self.out = nn.Sequential(
            normalization(ch, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )

    def to(self, *args, **kwargs):
        if isinstance(args[0], (list, tuple)) and len(args[0]) > 1:
            assert not kwargs and len(args) == 1
            self.devices = args[0]
            self.input_blocks.to(self.devices[0])
            self.time_embed.to(self.devices[0])
            self.middle_block.to(self.devices[0])
            if self.use_cross_attn:
                self.cond_encoder.to(self.devices[0])
            for k, b in enumerate(self.output_blocks):
                if k < self.decoder_device_thresh:
                    b.to(self.devices[0])
                else:
                    b.to(self.devices[1])
            self.out.to(self.devices[0])
            print(f"distributed UNet components to devices {self.devices}")
        else:
            super().to(*args, **kwargs)
            if self.devices is None:
                p = next(self.parameters())
                self.devices = [p.device, p.device]
        return self

    def forward(self, x, timesteps):
        """
        :param x:         [B, 32, H, W, D] — first 8ch = noisy target wavelet
                          subbands, last 24ch = condition subbands (3 modalities
                          x 8 subbands, the exact 3:1 ratio from cWDM).
        :param timesteps: [B] timestep indices.
        :return:          [B, 8, H, W, D] denoised target wavelet subbands.
        """
        hs = []
        input_pyramid = x
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = x

        # Run CondEncoder on condition subbands only: x[:, 8:, ...] = 24ch
        # This is the pure 3:1 condition signal — no target contamination.
        # feats: [finest, ..., coarsest] — reversed below for decoder order
        cond_feats_by_level = None
        if self.use_cross_attn:
            cond_input = x[:, 8:, ...]                   # [B, 24, H, W, D]
            raw_feats  = self.cond_encoder(cond_input)   # list: finest→coarsest
            # Reverse to coarsest→finest so index 0 matches deepest decoder level
            cond_feats_by_level = list(reversed(raw_feats))

        # Encoder (full 32ch input — cWDM conditioning intact)
        for module in self.input_blocks:
            if not isinstance(module[0], WaveletDownsample):
                h = module(h, emb)
                skip = None
                if isinstance(h, tuple):
                    h, skip = h
                hs.append(skip)
            else:
                input_pyramid = module(input_pyramid, emb)
                input_pyramid = input_pyramid + h
                h = input_pyramid

        # Middle block
        for module in self.middle_block:
            if isinstance(module, TimestepBlock):
                h = module(h, emb)
            else:
                h = module(h)
            if isinstance(h, tuple):
                h, skip = h

        # Decoder — supply matching CondEncoder features to each level
        # cond_feats_by_level[0] = coarsest (deepest decoder level)
        # cond_feats_by_level[-1] = finest (shallowest decoder level)
        blocks_per_level = self.num_res_blocks + 1
        cond_level_idx   = 0
        block_in_level   = 0

        for module in self.output_blocks:
            new_hs = hs.pop()

            if self.additive_skips:
                if new_hs is not None:
                    h = (h + new_hs) / np.sqrt(2)
            elif self.use_freq:
                if isinstance(h, tuple):
                    l = list(h); l[1] = new_hs; h = tuple(l)
                else:
                    h = (h, new_hs)
            else:
                if new_hs is not None:
                    h = th.cat([h, new_hs], dim=1)

            # Select CondEncoder features for this decoder level
            cf = None
            if cond_feats_by_level is not None and cond_level_idx < len(cond_feats_by_level):
                cf = cond_feats_by_level[cond_level_idx]

            h = module(h, emb, cond_feat=cf)

            block_in_level += 1
            if block_in_level == blocks_per_level:
                block_in_level = 0
                cond_level_idx += 1  # advance to the next finer scale

        for module in self.out_res:
            h = module(h, emb)

        h, _ = h
        return self.out(h)
