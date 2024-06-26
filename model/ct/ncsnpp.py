# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from . import layers, layerspp, normalization
import flax.linen as nn
import functools
import jax.numpy as jnp
import numpy as np
import omegaconf

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


class NCSNpp(nn.Module):
    """NCSN++ model"""

    # config: ml_collections.ConfigDict
    config: omegaconf.DictConfig

    @nn.compact
    def __call__(self, x, time_cond, train=True):
        # config parsing
        config = self.config
        act = get_act(config)
        nf = config["nf"]
        ch_mult = config["ch_mult"]
        num_res_blocks = config["num_res_blocks"]
        attn_resolutions = config["attn_resolutions"]
        dropout = config["dropout"]
        resamp_with_conv = config["resamp_with_conv"]
        num_resolutions = len(ch_mult)

        conditional = config["conditional"]  # noise-conditional
        fir = config["fir"]
        fir_kernel = config["fir_kernel"]
        skip_rescale = config["skip_rescale"]
        resblock_type = config["resblock_type"].lower()
        progressive = config["progressive"].lower()
        progressive_input = config["progressive_input"].lower()
        embedding_type = config["embedding_type"].lower()
        init_scale = config["init_scale"]
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = config["progressive_combine"].lower()
        combiner = functools.partial(Combine, method=combine_method)

        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            temb = layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config["fourier_scale"]
            )(time_cond)

        elif embedding_type == "positional":
            # Sinusoidal positional embeddings.
            temb = layers.get_timestep_embedding(time_cond, nf)
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional:
            temb = nn.Dense(nf * 4, kernel_init=default_initializer())(temb)
            temb = nn.Dense(nf * 4, kernel_init=default_initializer())(act(temb))
        else:
            temb = None

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive == "output_skip":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block

        input_pyramid = None
        if progressive_input != "none":
            input_pyramid = x

        hs = [conv3x3(x, nf)]
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(hs[-1], temb, train)
                if h.shape[1] in attn_resolutions:
                    h = AttnBlock()(h)
                hs.append(h)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    h = Downsample()(hs[-1])
                else:
                    h = ResnetBlock(down=True)(hs[-1], temb, train)

                if progressive_input == "input_skip":
                    input_pyramid = pyramid_downsample()(input_pyramid)
                    h = combiner()(input_pyramid, h)

                elif progressive_input == "residual":
                    input_pyramid = pyramid_downsample(out_ch=h.shape[-1])(
                        input_pyramid
                    )
                    if skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(
                            2.0, dtype=np.float32
                        )
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = ResnetBlock()(h, temb, train)
        h = AttnBlock()(h)
        h = ResnetBlock()(h, temb, train)

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                h = ResnetBlock(out_ch=nf * ch_mult[i_level])(
                    jnp.concatenate([h, hs.pop()], axis=-1), temb, train
                )

            if h.shape[1] in attn_resolutions:
                h = AttnBlock()(h)

            if progressive != "none":
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        pyramid = conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            x.shape[-1],
                            bias=True,
                            init_scale=init_scale,
                        )
                    elif progressive == "residual":
                        pyramid = conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            h.shape[-1],
                            bias=True,
                        )
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        pyramid = pyramid_upsample()(pyramid)
                        pyramid = pyramid + conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            x.shape[-1],
                            bias=True,
                            init_scale=init_scale,
                        )
                    elif progressive == "residual":
                        pyramid = pyramid_upsample(out_ch=h.shape[-1])(pyramid)
                        if skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0, dtype=np.float32)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    h = Upsample()(h)
                else:
                    h = ResnetBlock(up=True)(h, temb, train)

        assert not hs

        last_x_emb = None

        if progressive == "output_skip" and not config["double_heads"]:
            h = pyramid
        else:
            h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
            last_x_emb = h
            if config["double_heads"]:
                h = conv3x3(h, x.shape[-1] * 2, init_scale=init_scale)
            else:
                h = conv3x3(h, x.shape[-1], init_scale=init_scale)

        # return h
        return h, temb, last_x_emb


class JointNCSNpp(nn.Module):
    """NCSN++ model"""

    # config: ml_collections.ConfigDict
    config: omegaconf.DictConfig

    @nn.compact
    def __call__(self, x, time_cond, train=True):
        # config parsing
        denoiser = NCSNpp(config=self.config)
        distiller = NCSNpp(config=self.config)
        return denoiser(x, time_cond, train), distiller(x, time_cond, train)