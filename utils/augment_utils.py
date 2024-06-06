"""
This file is based on the official EDM implementation 
(https://github.com/NVlabs/edm/blob/main/training/augment.py)
The code is construted based on Pytorch, so I adjusted some features to adapt to JAX
"""

import jax
import jax.numpy as jnp

from functools import partial 

# Coefficient of various wavelet decomposition low-pass filters
wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}

_constant_cache = dict()

def constant(value, shape=None):
  if shape is not None:
    value = jnp.broadcast_to(value, shape)
  return jnp.asarray(value)

def matrix(*rows):
  assert all(len(row) == len(rows[0]) for row in rows)
  elems = [x for row in rows for x in row]
  ref = [x for x in elems if isinstance(x, jnp.ndarray)]
  if len(ref) == 0:
    return constant(jnp.array(rows))
  
  elems = [x if isinstance(x, jnp.ndarray) else constant(x, shape=ref[0].shape) for x in elems]
  # return constant(rows)
  return jnp.stack(elems, axis=-1).reshape(ref[0].shape + (len(rows), -1))

def translate2d(tx, ty, **kwargs):
  return matrix(
      [1, 0, tx],
      [0, 1, ty],
      [0, 0, 1],
      **kwargs)

def translate3d(tx, ty, tz, **kwargs):
  return matrix(
      [1, 0, 0, tx],
      [0, 1, 0, ty],
      [0, 0, 1, tz],
      [0, 0, 0, 1],
      **kwargs)

def scale2d(sx, sy, **kwargs):
  return matrix(
      [sx, 0,  0],
      [0,  sy, 0],
      [0,  0,  1],
      **kwargs)

def scale3d(sx, sy, sz, **kwargs):
  return matrix(
      [sx, 0,  0,  0],
      [0,  sy, 0,  0],
      [0,  0,  sz, 0],
      [0,  0,  0,  1],
      **kwargs)

def rotate2d(theta, **kwargs):
  return matrix(
      [jnp.cos(theta), jnp.sin(-theta), 0],
      [jnp.sin(theta), jnp.cos(theta),  0],
      [0,                0,                 1],
      **kwargs)

def rotate3d(v, theta, **kwargs):
  vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
  s = jnp.sin(theta); c = jnp.cos(theta); cc = 1 - c
  return matrix(
      [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
      [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
      [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
      [0,             0,             0,             1],
      **kwargs)

def translate2d_inv(tx, ty, **kwargs):
  return translate2d(-tx, -ty, **kwargs)

def scale2d_inv(sx, sy, **kwargs):
  return scale2d(1 / sx, 1 / sy, **kwargs)

def rotate2d_inv(theta, **kwargs):
  return rotate2d(-theta, **kwargs)

class AugmentPipe:
  def __init__(
        self, rng_key, p=1, xflip=0, yflip=0, rotate_int=0, translate_int=0,
        translate_int_max=0.125, scale=0, rotate_frac=0, aniso=0,
        translate_frac=0, scale_std=0.2, rotate_frac_max=1,
        aniso_std=0.2, aniso_rotate_prob=0.5, translate_frac_std=0.125,
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, 
        brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1
  ):
    super().__init__()
    self.rng_key = rng_key
    # self.images_shape = images_shape

    self.p                  = float(p)                  # Overall multiplier for augmentation probability.

    # Pixel blitting.
    self.xflip              = float(xflip)              # Probability multiplier for x-flip.
    self.yflip              = float(yflip)              # Probability multiplier for y-flip.
    self.rotate_int         = float(rotate_int)         # Probability multiplier for integer rotation.
    self.translate_int      = float(translate_int)      # Probability multiplier for integer translation.
    self.translate_int_max  = float(translate_int_max)  # Range of integer translation, relative to image dimensions.

    # Geometric transformations.
    self.scale              = float(scale)              # Probability multiplier for isotropic scaling.
    self.rotate_frac        = float(rotate_frac)        # Probability multiplier for fractional rotation.
    self.aniso              = float(aniso)              # Probability multiplier for anisotropic scaling.
    self.translate_frac     = float(translate_frac)     # Probability multiplier for fractional translation.
    self.scale_std          = float(scale_std)          # Log2 standard deviation of isotropic scaling.
    self.rotate_frac_max    = float(rotate_frac_max)    # Range of fractional rotation, 1 = full circle.
    self.aniso_std          = float(aniso_std)          # Log2 standard deviation of anisotropic scaling.
    self.aniso_rotate_prob  = float(aniso_rotate_prob)  # Probability of doing anisotropic scaling w.r.t. rotated coordinate frame.
    self.translate_frac_std = float(translate_frac_std) # Standard deviation of frational translation, relative to image dimensions.

    # Color transformations.
    self.brightness         = float(brightness)         # Probability multiplier for brightness.
    self.contrast           = float(contrast)           # Probability multiplier for contrast.
    self.lumaflip           = float(lumaflip)           # Probability multiplier for luma flip.
    self.hue                = float(hue)                # Probability multiplier for hue rotation.
    self.saturation         = float(saturation)         # Probability multiplier for saturation.
    self.brightness_std     = float(brightness_std)     # Standard deviation of brightness.
    self.contrast_std       = float(contrast_std)       # Log2 standard deviation of contrast.
    self.hue_max            = float(hue_max)            # Range of hue rotation, 1 = full circle.
    self.saturation_std     = float(saturation_std)     # Log2 standard deviation of saturation.
    
    def get_pixel_value_primitive(image, p_x, p_y):
      return image[:, p_y, p_x]
    
    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=1)
    @jax.jit
    def get_pixel_value_3rd_fn(images, x, y):
      return get_pixel_value_primitive(images, x, y)

    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=1)
    @jax.jit
    def get_pixel_value_2nd_fn(images, x, y):
      return get_pixel_value_3rd_fn(images, x, y)

    @jax.jit
    def get_pixel_value_fn(images, x, y):
      return get_pixel_value_2nd_fn(images, x, y)

    self.get_pixel_value_vmap = jax.vmap(get_pixel_value_fn)

  # @partial(jax.jit, static_argnums=(0,))
  def affine_grid(self, theta, shape):
    """
    Reference:
    https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py
    """
    theta = jnp.reshape(theta, (-1, 2, 3))
    b, c, h, w = shape
    x = jnp.linspace(-1.0, 1.0, w)
    y = jnp.linspace(-1.0, 1.0, h)

    x_t, y_t = jnp.meshgrid(x, y)

    x_t_flat = jnp.reshape(x_t, [-1])
    y_t_flat = jnp.reshape(y_t, [-1])

    ones = jnp.ones_like(x_t_flat)

    # Grid with ones
    grid = jnp.stack([x_t_flat, y_t_flat, ones])
    grid = jnp.expand_dims(grid, axis=0)
    batch_grid = jnp.repeat(grid, b, axis=0)

    theta = theta.astype(jnp.float32)
    batch_grid = batch_grid.astype(jnp.float32)

    batch_grid = jnp.matmul(theta, batch_grid)
    batch_grid = jnp.reshape(batch_grid, [b, 2, h, w])
    return batch_grid

  def get_pixel_value(self, image, x, y):
    shape = jnp.shape(x)
    batch_size = shape[0]

    batch_idx = jnp.arange(0, batch_size)
    batch_idx = jnp.reshape(batch_idx, (batch_size, 1, 1))

    result_value = self.get_pixel_value_vmap(image, x, y)
    return result_value

  @partial(jax.jit, static_argnums=(0,))
  def grid_sampler(self, image, grid):
    x = grid[:, 0, :, :]
    y = grid[:, 1, :, :]

    H = image.shape[2]
    W = image.shape[3]

    max_x = jnp.asarray(W-1, dtype=jnp.int32)
    max_y = jnp.asarray(H-1, dtype=jnp.int32)
    zero = jnp.zeros([], dtype=jnp.int32)

    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    x = 0.5 * ((x + 1.0) * (max_x - 1).astype(jnp.float32))
    y = 0.5 * ((y + 1.0) * (max_y - 1).astype(jnp.float32))

    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = y0 + 1

    x0 = jnp.clip(x0, zero, max_x)
    x1 = jnp.clip(x1, zero, max_x)
    y0 = jnp.clip(y0, zero, max_y)
    y1 = jnp.clip(y1, zero, max_y)

    la = self.get_pixel_value(image, x0, y0)
    lb = self.get_pixel_value(image, x0, y1)
    lc = self.get_pixel_value(image, x1, y0)
    ld = self.get_pixel_value(image, x1, y1)

    x0 = x0.astype(jnp.float32)
    x1 = x1.astype(jnp.float32)
    y0 = y0.astype(jnp.float32)
    y1 = y1.astype(jnp.float32)

    wa = jnp.expand_dims((x1 - x) * (y1 - y), axis=1)
    wb = jnp.expand_dims((x1 - x) * (y - y0), axis=1)
    wc = jnp.expand_dims((x - x0) * (y1 - y), axis=1)
    wd = jnp.expand_dims((x - x0) * (y - y0), axis=1)
    
    out = wa * la + wb * lb + wc * lc + wd * ld
    return out
    
  @partial(jax.jit, static_argnums=(0,))
  def xflip_fn(self, key, images):
    B = images.shape[0]
    key, xflip_key1, xflip_key2 = jax.random.split(key, 3)
    w = jax.random.randint(xflip_key1, shape=(B, 1, 1, 1), minval=0, maxval=2)
    w = jnp.where(
      jax.random.uniform(xflip_key2, shape=(B, 1, 1, 1)) < self.xflip * self.p,
      w, jnp.zeros_like(w)
    )
    images = jnp.where(w == 1, jnp.flip(images, 3), images)
    return images, [w]
  
  @partial(jax.jit, static_argnums=(0,))
  def yflip_fn(self, key, images):
    B = images.shape[0]
    key, yflip_key1, yflip_key2 = jax.random.split(key, 3)
    w = jax.random.randint(yflip_key1, shape=(B, 1, 1, 1), minval=0, maxval=2)
    w = jnp.where(
      jax.random.uniform(yflip_key2, shape=(B, 1, 1, 1)) < self.yflip * self.p,
      w, jnp.zeros_like(w)
    )
    images = jnp.where(w == 1, jnp.flip(images, 2), images)
    return images, [w]
  
  @partial(jax.jit, static_argnums=(0,))
  def rotate_int_fn(self, key, images):
    B = images.shape[0]
    key, rotate_key1, rotate_key2 = jax.random.split(key, 3)
    w = jax.random.randint(rotate_key1, shape=(B, 1, 1, 1), minval=0, maxval=4)
    w = jnp.where(
      jax.random.uniform(rotate_key2, (B, 1, 1, 1)) < self.rotate_int * self.p,
      w, jnp.zeros_like(w)
    )
    images = jnp.where((w == 1) | (w == 2), jnp.flip(images, 3), images)
    images = jnp.where((w == 2) | (w == 3), jnp.flip(images, 2), images)
    images = jnp.where((w == 1) | (w == 3), jnp.transpose(images, (2, 3)), images)
    return images, [(w == 1) | (w == 2), (w == 2) | (w == 3)]

  @partial(jax.jit, static_argnums=(0,))
  def translate_int_fn(self, key, images):
    B, H, W, C = images.shape
    key, translate_key1, translate_key2 = jax.random.split(key, 3)
    w = jax.random.uniform(translate_key1, (2, B, 1, 1, 1)) * 2 - 1
    w = jnp.where(
      jax.random.uniform(translate_key2, (1, B, 1, 1, 1)) < self.translate_int * self.p,
      w,
      jnp.zeros_like(w)
    )
    tx = jnp.round(w[0] * (W * self.translate_int_max))
    ty = jnp.round(w[1] * (H * self.translate_int_max))
    b, c, y, x = jnp.meshgrid(*(jnp.arange(x) for x in images.shape), indexing='ij')
    x = W - 1 - jnp.abs((W - 1 - (x - tx) % (W * 2 - 2)))
    y = H - 1 - jnp.abs((H - 1 - (y + ty) % (H * 2 - 2)))
    images = images.flatten()[(((b * C) + c) * H + y) * W + x]
    return images, [tx / (W * self.translate_int_max), ty / (H * self.translate_int_max)]
  
  @partial(jax.jit, static_argnums=(0,))
  def scale_fn(self, key, images, G_inv):
    B = images.shape[0]
    key, scale_key1, scale_key2 = jax.random.split(key, 3)
    w = jax.random.normal(scale_key1, (B,))
    w = jnp.where(
      jax.random.uniform(scale_key2, (B,)) < self.scale * self.p,
      w, jnp.zeros_like(w))
    s = 2 ** (w * self.scale_std)
    G_inv = G_inv @ scale2d_inv(s, s)
    return G_inv, [w]
  
  @partial(jax.jit, static_argnums=(0,))
  def rotate_fn(self, key, images, G_inv):
    B= images.shape[0]
    key, rotate_key1, rotate_key2 = jax.random.split(key, 3)
    w = (jax.random.uniform(rotate_key1, (B,)) * 2 - 1) * (jnp.pi * self.rotate_frac_max)
    w = jnp.where(
      jax.random.uniform(rotate_key2, (B,)) < self.rotate_frac * self.p, 
      w, jnp.zeros_like(w))
    G_inv = G_inv @ rotate2d_inv(-w)
    return G_inv, [jnp.cos(w) - 1, jnp.sin(w)]

  @partial(jax.jit, static_argnums=(0,))
  def aniso_fn(self, key, images, G_inv):
    B= images.shape[0]
    key, aniso_key1, aniso_key2, aniso_key3, aniso_key4 = jax.random.split(key, 5)
    w = jax.random.normal(aniso_key1, (B,))
    r = (jax.random.uniform(aniso_key2, (B,)) * 2 - 1) * jnp.pi
    w = jnp.where(
      jax.random.uniform(aniso_key3, (B,)) < self.aniso * self.p, 
      w, jnp.zeros_like(w))
    r = jnp.where(
      jax.random.uniform(aniso_key4, (B,)) < self.aniso_rotate_prob, 
      r, jnp.zeros_like(r))
    s = 2 ** (w * self.aniso_std)
    G_inv = G_inv @ rotate2d_inv(r) @ scale2d_inv(s, 1 / s) @ rotate2d_inv(-r)
    return G_inv, [w * jnp.cos(r), w * jnp.sin(r)]
  
  @partial(jax.jit, static_argnums=(0,))
  def translate_frac_fn(self, key, images, G_inv):
    B = images.shape[0]
    H = images.shape[2]
    W = images.shape[3]
    
    key, tf_key1, tf_key2 = jax.random.split(key, 3)
    w = jax.random.normal(tf_key1, [2, B])
    w = jnp.where(
      jax.random.uniform(tf_key2, [1, B]) < self.translate_frac * self.p, 
      w, jnp.zeros_like(w))
    G_inv = G_inv @ translate2d_inv(w[0] * W * self.translate_frac_std, w[1] * H * self.translate_frac_std)
    return G_inv, [w[0], w[1]]

  # @partial(jax.jit, static_argnums=(0,))
  @partial(jax.pmap, static_broadcasted_argnums=(0,))
  def before_apply_padding_geometry(self, G_inv, images):
    W, H = images.shape[-1], images.shape[-2]
    cx = (W - 1) / 2
    cy = (H - 1) / 2
    cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1]) # [idx, xyz]
    cp = G_inv @ cp.T # [batch, xyz, idx]
    Hz = jnp.asarray(wavelets['sym6'], dtype=jnp.float32)
    Hz_pad = len(Hz) // 4

    margin = jnp.transpose(cp[:, :2, :], (1, 0, 2))
    margin = jnp.reshape(margin, (margin.shape[0], -1)) # [xy, batch * idx]
    margin = jnp.concatenate([-margin, margin]).max(axis=1) # [x0, y0, x1, y1]
    margin = margin + constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2)
    margin = jnp.maximum(margin, constant([0, 0] * 2))
    margin = jnp.minimum(margin, constant([W - 1, H - 1] * 2))
    mx0, my0, mx1, my1 = jnp.ceil(margin).astype(jnp.int32)
    return mx0, my0, mx1, my1

  # @partial(jax.jit, static_argnums=(0,))
  @partial(jax.pmap, static_broadcasted_argnums=(0,))
  def after_apply_padding_geometry(self, images, G_inv, padding, original_image_dummy):
    image_size = original_image_dummy.shape
    B, C, H, W = image_size[0], image_size[1], image_size[2], image_size[3]

    mx0, my0, mx1, my1 = padding[0], padding[1], padding[2], padding[3]
    G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

    Hz = jnp.asarray(wavelets['sym6'], dtype=jnp.float32)
    Hz_pad = len(Hz) // 4

    # Upsample.
    conv_weight = jnp.tile(constant(Hz[None, None, ::-1]), [images.shape[1], 1, 1])
    conv_pad = (len(Hz) + 1) // 2
    images = jnp.stack([images, jnp.zeros_like(images)], axis=4).reshape(B, C, images.shape[2], -1)[:, :, :, :-1]
    images = jax.lax.conv_general_dilated(images, jnp.expand_dims(conv_weight, 2), 
                                          (1, 1), padding=[[0, 0], [conv_pad, conv_pad]], feature_group_count=images.shape[1], 
                                          dimension_numbers=("NCHW", "OIHW", "NCHW"))
    images = jnp.stack([images, jnp.zeros_like(images)], axis=3).reshape(B, C, -1, images.shape[3])[:, :, :-1, :]
    images = jax.lax.conv_general_dilated(images, jnp.expand_dims(conv_weight, 3), 
                                          (1, 1), padding=[[conv_pad, conv_pad], [0, 0]], feature_group_count=images.shape[1], 
                                          dimension_numbers=("NCHW", "OIHW", "NCHW"))
    G_inv = scale2d(2, 2) @ G_inv @ scale2d_inv(2, 2)
    G_inv = translate2d(-0.5, -0.5) @ G_inv @ translate2d_inv(-0.5, -0.5)

    # Execute transformation.
    shape = [B, C, (H + Hz_pad * 2) * 2, (W + Hz_pad * 2) * 2]
    G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2]) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2])
    grid = self.affine_grid(theta=G_inv[:,:2,:], shape=shape)
    images = self.grid_sampler(images, grid)

    # Downsample and crop.
    conv_weight = jnp.tile(constant(Hz[None, None, :]), [images.shape[1], 1, 1])
    conv_pad = (len(Hz) - 1) // 2
    images = jax.lax.conv_general_dilated(images, jnp.expand_dims(conv_weight, 2), 
                                          (1, 2), padding=[[0, 0], [conv_pad, conv_pad]], feature_group_count=images.shape[1], 
                                          dimension_numbers=("NCHW", "OIHW", "NCHW"))[:, :, :, Hz_pad : -Hz_pad]
    images = jax.lax.conv_general_dilated(images, jnp.expand_dims(conv_weight, 3), 
                                          (2, 1), padding=[[conv_pad, conv_pad], [0, 0]], feature_group_count=images.shape[1], 
                                          dimension_numbers=("NCHW", "OIHW", "NCHW"))[:, :, Hz_pad : -Hz_pad, :]

    return images

  @partial(jax.pmap, static_broadcasted_argnums=(0,))
  def pixel_blitting(self, rng, images, labels):
    augmentation_key = rng
    if self.xflip > 0:
      augmentation_key, tmp_key = jax.random.split(augmentation_key)
      images, add_labels = self.xflip_fn(tmp_key, images)
      labels += add_labels
    
    if self.yflip > 0:
      augmentation_key, tmp_key = jax.random.split(augmentation_key)
      images, add_labels = self.yflip_fn(tmp_key, images)
      labels += add_labels

    if self.rotate_int > 0:
      augmentation_key, tmp_key = jax.random.split(augmentation_key)
      images, add_labels = self.rotate_int_fn(tmp_key, images)
      labels += add_labels
    
    if self.translate_int > 0:
      augmentation_key, tmp_key = jax.random.split(augmentation_key)
      images, add_labels = self.translate_int_fn(tmp_key, images)
      labels += add_labels
    return images, labels

  @partial(jax.pmap, static_broadcasted_argnums=(0,))
  def geometric_transform(self, rng, images, labels):
    augmentation_key = rng
    I_3 = jnp.eye(3)
    G_inv = I_3

    if self.scale > 0:
      augmentation_key, tmp_key = jax.random.split(augmentation_key)
      G_inv, add_labels = self.scale_fn(tmp_key, images, G_inv)
      labels += add_labels
    
    if self.rotate_frac > 0:
      augmentation_key, tmp_key = jax.random.split(augmentation_key)
      G_inv, add_labels = self.rotate_fn(tmp_key, images, G_inv)
      labels += add_labels

    if self.aniso > 0:
      augmentation_key, tmp_key = jax.random.split(augmentation_key)
      G_inv, add_labels = self.aniso_fn(tmp_key, images, G_inv)
      labels += add_labels
    
    if self.translate_frac > 0:
      augmentation_key, tmp_key = jax.random.split(augmentation_key)
      G_inv, add_labels = self.translate_frac_fn(tmp_key, images, G_inv)
      labels += add_labels
    return images, labels, G_inv
  
  @partial(jax.pmap, static_broadcasted_argnums=(0,))
  def color_transform(self, rng, images, labels):
    augmentation_key = rng
    I_4 = jnp.eye(4)
    M = I_4
    B = images.shape[0]

    luma_axis = constant(jnp.asarray([1, 1, 1, 0]) / jnp.sqrt(3))
    if self.brightness > 0:
      augmentation_key, brightness_key1, brightness_key2 = jax.random.split(augmentation_key, 3)
      w = jax.random.normal(brightness_key1, (B,))
      w = jnp.where(jax.random.uniform(brightness_key2, (B,)) < self.brightness * self.p, w, jnp.zeros_like(w))
      b = w * self.brightness_std
      M = translate3d(b, b, b) @ M
      labels += [w]

    if self.contrast > 0:
      augmentation_key, contrast_key1, contrast_key2 = jax.random.split(augmentation_key, 3)
      w = jax.random.normal(contrast_key1, (B,))
      w = jnp.where(jnp.random.uniform(contrast_key2, (B,)) < self.contrast * self.p, w, jnp.zeros_like(w))
      c = 2 ** (w * self.contrast_std)
      M = scale3d(c, c, c) @ M
      labels += [w]

    if self.lumaflip > 0:
      augmentation_key, lumaflip_key1, lumaflip_key2 = jax.random.split(augmentation_key, 3)
      w = jax.random.randint(lumaflip_key1, (B, 1, 1), 0, 2)
      w = jnp.where(jax.random.uniform(lumaflip_key2, (B, 1, 1)) < self.lumaflip * self.p, w, jnp.zeros_like(w))
      M = (I_4 - 2 * jnp.outer(luma_axis, luma_axis) * w) @ M
      labels += [w]

    if self.hue > 0:
      augmentation_key, hue_key1, hue_key2 = jax.random.split(augmentation_key, 3)
      w = (jax.random.uniform(hue_key1, (B,)) * 2 - 1) * (jnp.pi * self.hue_max)
      w = jnp.where(jax.random.uniform(hue_key2, (B,)) < self.hue * self.p, w, jnp.zeros_like(w))
      M = rotate3d(luma_axis, w) @ M
      labels += [jnp.cos(w) - 1, jnp.sin(w)]

    if self.saturation > 0:
      augmentation_key, saturation_key1, saturation_key2 = jax.random.split(augmentation_key, 3)
      w = jax.random.normal(saturation_key1, (B, 1, 1))
      w = jnp.where(jax.random.uniform(saturation_key2, (B, 1, 1)) < self.saturation * self.p, w, jnp.zeros_like(w))
      M = (jnp.outer(luma_axis, luma_axis) + (I_4 - jnp.outer(luma_axis, luma_axis)) * (2 ** (w * self.saturation_std))) @ M
      labels += [w]
    return M, labels
  
  @partial(jax.pmap, static_broadcasted_argnums=(0,))
  def apply_color_transform(self, images, M):
    # B, C, H, W = images.shape()
    image_size = images.shape
    B, C, H, W = image_size[0], image_size[1], image_size[2], image_size[3]

    images = images.reshape([B, C, H * W])
    if C == 3:
        images = M[:, :3, :3] @ images + M[:, :3, 3:]
    elif C == 1:
        M = M[:, :3, :].mean(dim=1, keepdims=True)
        images = images * jnp.sum(M[:, :, :3], axis=2, keepdims=True)+ M[:, :, 3:]
    else:
        raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
    images = images.reshape([B, C, H, W])
    return images
    
  def __call__(self, images):
    pmap=False
    if len(images.shape) != 4:
      pmap=True
      original_images_format = images.shape
      images= images.reshape((-1, *images.shape[-3:]))

    B, H, W, C = images.shape

    images = jnp.transpose(images, (0, 3, 1, 2))
    
    images = images.reshape(jax.local_device_count(), B // jax.local_device_count(), C, H, W)
    labels = [jnp.zeros([jax.local_device_count(), B // jax.local_device_count(), 0])]

    self.rng_key, augmentation_key = jax.random.split(self.rng_key)
    augmentation_key = jax.random.split(augmentation_key, jax.local_device_count())
    images, labels = self.pixel_blitting(augmentation_key, images, labels)

    # Select parameters for geometric translations
    self.rng_key, augmentation_key = jax.random.split(self.rng_key)
    augmentation_key = jax.random.split(augmentation_key, jax.local_device_count())
    images, labels, G_inv = self.geometric_transform(augmentation_key, images, labels)
    
    
    # Execute geometric transformations
    if not jnp.all(G_inv == jnp.eye(3)):
      original_images_size_dummy = jnp.zeros(images.shape)
      mx0, my0, mx1, my1 = self.before_apply_padding_geometry(G_inv, images)

      mx0 = jnp.max(mx0)
      my0 = jnp.max(my0)
      mx1 = jnp.max(mx1)
      my1 = jnp.max(my1)

      # Pad image and adjust origin.
      padding = [[0, 0], [0, 0], [0, 0], [my0, my1], [mx0, mx1]]
      images = jnp.pad(images, padding, mode='reflect')

      padding = jnp.asarray([[mx0, my0, mx1, my1]] * jax.local_device_count())
      images = self.after_apply_padding_geometry(images, G_inv, padding, original_images_size_dummy)

    self.rng_key, augmentation_key = jax.random.split(self.rng_key)
    augmentation_key = jax.random.split(augmentation_key, jax.local_device_count())
    M, labels = self.color_transform(augmentation_key, images, labels)

    # Execute color transformations
    if not jnp.all(M == jnp.eye(4)):
      images = self.apply_color_transform(images, M)

    labels = jnp.concatenate([x.astype(jnp.float32).reshape(B, -1) for x in labels], axis=1)
    images = jnp.reshape(images, (B, *images.shape[2:]))
    images = jnp.transpose(images, (0, 2, 3, 1))
    if pmap:
      images = images.reshape(*original_images_format)
      labels = labels.reshape(*original_images_format[:-3], -1)
    return images, labels


if __name__ == "__main__":
  import common_utils
  import matplotlib.pyplot as plt
  import os 
  import numpy as np

  datasets = common_utils.load_dataset_from_tfds(pmap=True)
  augment_rng = jax.random.PRNGKey(0)
  augment_rate = 0.12
  pipeline = AugmentPipe(
    augment_rng, p=augment_rate, xflip=1e8, 
    yflip=1, scale=1, rotate_frac=1, 
    aniso=1, translate_frac=1)
  
  def tmp_save_comparison(images, steps, savepath):
    # Make in process dir first
    # self.verify_and_create_dir(savepath)
    savepath="."

    images = common_utils.unnormalize_minus_one_to_one(images)
    n_images = len(images)
    f, axes = plt.subplots(n_images // 4, 4)
    images = np.clip(images, 0, 1)
    axes = np.concatenate(axes)

    for img, axis in zip(images, axes):
        axis.imshow(img)
        axis.axis('off')
    
    save_filename = os.path.join(savepath, f"{steps}.png")
    f.savefig(save_filename)
    plt.close()
    return save_filename

  idx = 0
  for x, _, in datasets:
    result_images, labels = pipeline(x)
    result_images = result_images[0, :8]
    x = x[0, :8]
    xset = jnp.concatenate([result_images[:8], x], axis=0)
    sample_path = tmp_save_comparison(xset, idx, ".")
    idx += 1
    breakpoint()