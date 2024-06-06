import jax
import jax.numpy as jnp
import flax.linen as nn
import math

from functools import partial

from typing import Sequence, Union, Tuple

from einops import rearrange

def create_initializer(init_name: str = None):
    def get_in_out_features(shape, fan_in, fan_out):
        if fan_in is None and fan_out is None:
            if len(shape) == 4:
                in_feature= shape[0] * shape[1] * shape[2]
                out_feature= shape[0] * shape[1] * shape[3]
            else:
                in_feature, out_feature = shape[-2:]
        else:
            in_feature, out_feature = fan_in, fan_out
        return in_feature, out_feature

    def xavier_uniform(key: jax.random.PRNGKey, shape, dtype=jnp.float32, fan_in=None, fan_out=None):
        in_feature, out_feature = get_in_out_features(shape, fan_in=fan_in, fan_out=fan_out)
        rand_shape = jax.random.uniform(key, shape)
        return_value = jnp.sqrt(6 / (in_feature + out_feature)) * (rand_shape * 2 - 1)
        return_value = return_value.astype(dtype)
        return return_value
    
    def kaiminig_uniform(key: jax.random.PRNGKey, shape, dtype=jnp.float32, fan_in=None, fan_out=None, mode="fan_avg", scale=1.0):
        in_feature, out_feature = get_in_out_features(shape, fan_in=fan_in, fan_out=fan_out)
        if mode == "fan_avg":
            n = (in_feature + out_feature) / 2
        elif mode == "fan_in":
            n = in_feature
        elif mode == "fan_out":
            n = out_feature
        else:
            NotImplementedError("Kaiming uniform can only take 'fan_avg', 'fan_in', and 'fan_out'.")
        
        effective_scale = 1e-10 if scale == 0 else scale
        limit = jnp.sqrt(3 * effective_scale / n)
        rand_shape = jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        return_value = rand_shape.astype(dtype)
        return return_value

    if init_name is None:
        return None
    
    elif init_name == "xavier_uniform":
        return xavier_uniform

    elif init_name == "xavier_zero":
        def xavier_zero(key: jax.random.PRNGKey, shape, dtype=jnp.float32, fan_in=None, fan_out=None):
            return xavier_uniform(key, shape, dtype, fan_in, fan_out) * 1e-5 
        return xavier_zero

    elif init_name == "xavier_attn":
        def xavier_attn(key: jax.random.PRNGKey, shape, dtype=jnp.float32, fan_in=None, fan_out=None):
            return xavier_uniform(key, shape, dtype, fan_in, fan_out) * jnp.sqrt(0.2)
        return xavier_attn

    elif init_name == "kaiming_uniform":
        # return kaiminig_uniform
        return jax.nn.initializers.variance_scaling(1.0, "fan_avg", "uniform")

    elif init_name == "kaiming_zero":
        # return partial(kaiminig_uniform, scale=0)
        return jax.nn.initializers.variance_scaling(1e-10, "fan_avg", "uniform")

    else:
        NotImplementedError(f"{init_name} initializer is not supported.")

class CustomConv2d(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: Union[int, int, Sequence[int]] = 1
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "SAME"
    use_bias: bool = True
    init_scale: float = 1.0
    
    @nn.compact
    def __call__(self, x):
        init_mode = "kaiming_zero" if self.init_scale == 0 else "kaiming_uniform"
        init_function = create_initializer(init_mode)
        x = nn.Conv(features=self.features,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    use_bias=self.use_bias,
                    kernel_init=init_function)(x)
        return x

class CustomDense(nn.Module):
    features: int
    use_bias: bool = True
    init_scale: float = 1.0
    
    @nn.compact
    def __call__(self, x):
        init_mode = "kaiming_zero" if self.init_scale == 0 else "kaiming_uniform"
        init_function = create_initializer(init_mode)
        x = nn.Dense(features=self.features,
                    use_bias=self.use_bias,
                    kernel_init=init_function)(x)
        return x

class TimeEmbedding(nn.Module):
    emb_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, time):
        assert len(time.shape) == 1
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=self.dtype) * -emb)
        emb = time.astype(self.dtype)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        return emb

class TimeEmbed(nn.Module):
    n_channels: int
    hidden_size: int
    activation_type: str = "swish"
    
    @nn.compact
    def __call__(self, t):
        t = TimeEmbedding(self.n_channels)(t)
        t = CustomDense(self.hidden_size)(t)
        t = getattr(nn, self.activation_type)(t)
        t = CustomDense(self.hidden_size)(t)
        return t
    

class ResidualBlock(nn.Module):
    out_channels: int
    n_groups: int = 32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, t, train):
        h = nn.GroupNorm(self.n_groups)(x)
        h = nn.swish(h)
        h = CustomConv2d(self.out_channels, (3, 3))(h)

        # Add time embedding value
        if t is not None:
            t = nn.swish(t)
            t_emb = CustomDense(self.out_channels)(t)
            h += t_emb[:, None, None, :]

        h = nn.GroupNorm(self.n_groups)(h)
        h = nn.swish(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not train)(h)
        h = CustomConv2d(self.out_channels, (3, 3), init_scale=0.)(h)

        if x.shape != h.shape:
            # short = CustomConv2d(self.out_channels, (3, 3))(x)
            short = CustomConv2d(self.out_channels, (1, 1))(x)
        else:
            # short = CustomDense(self.out_channels)(x)
            short = x
        return h + short

class AttentionBlock(nn.Module):
    n_channels: int
    n_heads: int = 1
    n_groups: int = 32
    
    @nn.compact
    def __call__(self, x): # x: b x y c
        # scale = self.n_channels ** -0.5

        batch_size, height, width, n_channels = x.shape
        head_channels = n_channels // self.n_heads
        scale = head_channels ** -0.5
        # Projection
        x_skip = x
        x = nn.GroupNorm(self.n_groups)(x)
        qkv = CustomConv2d(self.n_heads * head_channels * 3, (1, 1), use_bias=False)(x) # qkv: b x y h*c*3
        qkv = rearrange(qkv, "b y x (h c) -> b (y x) h c", h=self.n_heads)

        # Split as query, key, value
        q, k, v = jnp.split(qkv, 3, axis=-1) # q,k,v = b (x y) h c

        q = q * scale

        # Scale dot product 
        atten = jnp.einsum('b i h d, b j h d -> b i j h', q, k)

        # Softmax
        atten = nn.softmax(atten, axis=2)

        # Multiply by value
        res = jnp.einsum('b i j h, b j h d -> b i h d', atten, v)
        # res = jnp.einsum('b i j h, b i h d -> b j h d', atten, v)

        res = rearrange(res, "b (y x) h c-> b y x (h c)", x=width, y=height)
        res = CustomConv2d(self.n_channels, (1, 1), init_scale=0.)(res)

        # skip connection
        res += x_skip

        return res
# class AttentionBlock(nn.Module):
#     n_channels: int
#     n_heads: int = 1
#     n_groups: int = 8
    
#     @nn.compact
#     def __call__(self, x): # x: b x y c
#         scale = self.n_channels ** -0.5

#         batch_size, height, width, n_channels = x.shape
#         head_channels = n_channels // self.n_heads
#         # Projection
#         x_skip = x
#         x = nn.GroupNorm(self.n_groups)(x)
#         qkv = nn.Conv(self.n_heads * head_channels * 3, (1, 1), use_bias=False)(x) # qkv: b x y h*c*3
#         qkv = qkv.reshape(batch_size, -1, self.n_heads, 3 * head_channels) # b (x y) h c*3

#         # Split as query, key, value
#         q, k, v = jnp.split(qkv, 3, axis=-1) # q,k,v = b (x y) h c

#         # Scale dot product 
#         atten = jnp.einsum('bihd,bjhd->bijh', q, k) * scale

#         # Softmax
#         atten = nn.softmax(atten, axis=2)

#         # Multiply by value
#         res = jnp.einsum('bijh,bjhd->bihd', atten, v)

#         # res = res.reshape(batch_size, -1, self.n_heads * self.n_channels)
#         res = res.reshape(batch_size, height, width, self.n_heads * head_channels)
#         # res = nn.Dense(n_channels)(res)
#         res = nn.Conv(self.n_channels, (1, 1))(res)

#         # skip connection
#         res += x_skip

#         return res


class UnetDown(nn.Module):
    out_channels: int
    has_atten: bool
    dropout_rate: float
    n_groups: int
    
    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.out_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        if self.has_atten:
            x = AttentionBlock(self.out_channels, n_groups=self.n_groups)(x)
        return x

class UnetUp(nn.Module):
    out_channels: int
    has_atten: bool
    dropout_rate: float
    n_groups: int
    
    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.out_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        if self.has_atten:
            x = AttentionBlock(self.out_channels, n_groups=self.n_groups)(x)
        return x

class UnetMiddle(nn.Module):
    n_channels: int
    dropout_rate: float
    n_groups: int

    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.n_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        x = AttentionBlock(self.n_channels)(x)
        x = ResidualBlock(self.n_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        return x


class Upsample(nn.Module):
    n_channels: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        scale = 2
        x = jax.image.resize(
            x, 
            shape=(B, H * scale, W * scale, C),
            method="nearest")
        x = CustomConv2d(self.n_channels, (3, 3))(x)
        return x

class Downsample(nn.Module):
    n_channels: int

    @nn.compact
    def __call__(self, x):
        # B, H, W, C = x.shape
        # x = jnp.reshape(x, (B, H // 2, W // 2, C * 4))
        x = CustomConv2d(self.n_channels, (3, 3), strides=2)(x)
        return x


class VectorQuantizer(nn.Module):
    """
    Original github: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
    This model is ported from pytorch to jax, flax.

    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    n_e: int
    e_dim: int
    beta: float
    remap: str = None
    unknown_index: str = "random"
    sane_index_shape: bool = False
    legacy: bool = True

    def setup(self):
        def zero_centered_uniform(key, shape, dtype=jnp.float_):
            scale = 1.0 / self.n_e
            data = jax.random.uniform(key, shape, minval=-scale, maxval=scale)
            return data
        self.embedding = nn.Embed(self.n_e, self.e_dim, embedding_init=zero_centered_uniform)

    def __call__(self, z): # z: b h w c
        z_flatten = jnp.reshape(z, (-1, self.e_dim)) # b*h*w*c/3, 3
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        embedding = self.embedding.embedding
        d = jnp.sum(z_flatten ** 2, axis=1, keepdims=True) + \
            jnp.sum(embedding ** 2, axis=1) - \
            2 * jnp.einsum('bd, dn -> bn', z_flatten, jnp.transpose(embedding))
        min_encoding_indices = jnp.argmin(d, axis=1)
        z_q = self.embedding(min_encoding_indices)
        z_q = jnp.reshape(z_q, z.shape)
        perplexity = None
        min_encodings = None
        loss = jnp.mean((jax.lax.stop_gradient(z_q) - z) ** 2) + \
                jnp.mean((z_q - jax.lax.stop_gradient(z)) ** 2)
        
        # Preserve gradients
        z_q = z + jax.lax.stop_gradient(z_q - z)

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

## For DiT
class LabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int
    dropout_prob: float
    
    def token_drop(self, labels, force_drop_ids=None):
        """
        Token drop function for classifier free guidence (although it is not used for now.)
        """
        if force_drop_ids is None:
            rng_val = self.make_rng('CFG') # TODO: the models should contain additional PRNG
            drop_ids = jax.random.uniform(rng_val, labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embed_classes = self.num_classes + 1 if self.dropout_prob > 0 else self.num_classes
        embeddings = nn.Embed(embed_classes, self.hidden_size)(labels)
        return embeddings

class FeedForwardMLP(nn.Module):
    hidden_features: int
    out_features: int
    dropout_rate: float
    act_layer: str = "gelu"
    
    @nn.compact
    def __call__(self, x, train):
        x = nn.Dense(self.hidden_features)(x)
        x = getattr(nn, self.act_layer)(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.out_features)(x)
        x = getattr(nn, self.act_layer)(x)
        return x

class PatchEmbed(nn.Module):
    img_size: int
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 128
    norm_layer: bool = False
    flatten: bool = True
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        assert H == W
        assert H == self.img_size
        x = nn.Conv(
            self.embed_dim, 
            (self.patch_size, self.patch_size), 
            self.patch_size,
            use_bias=self.bias)(x)
        if self.flatten:
            x = jnp.reshape(x, (B, -1, self.embed_dim))
        
        if self.norm_layer:
            x = nn.LayerNorm()(x)
        return x
            