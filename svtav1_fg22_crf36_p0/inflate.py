#!/usr/bin/env python
import os, av, torch
import torch.nn.functional as F
from frame_utils import camera_size, yuv420_to_rgb

# decode-time settings, picked with a local parameter sweep
UPSCALE_MODE = 'bicubic'
SHARPEN = 0.0  # unsharp mask amount applied after upscaling, 0 disables


def pick_device():
  if os.environ.get('INFLATE_DEVICE'): return torch.device(os.environ['INFLATE_DEVICE'])
  if torch.cuda.is_available(): return torch.device('cuda')
  if torch.backends.mps.is_available(): return torch.device('mps')
  return torch.device('cpu')


def _gaussian_blur(x, sigma=1.0):
  k = int(2 * round(3 * sigma) + 1)
  t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
  g = torch.exp(-t ** 2 / (2 * sigma ** 2))
  g = g / g.sum()
  C = x.shape[1]
  x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode='replicate'), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
  return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode='replicate'), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def upscale(frames, mode=UPSCALE_MODE, sharpen=SHARPEN):
  """(B, h, w, 3) uint8 -> (B, H, W, 3) uint8 at camera resolution."""
  target_w, target_h = camera_size
  if mode == 'stretch':
    return _stretch(frames, target_h, target_w) if not sharpen else upscale(_stretch(frames, target_h, target_w), 'bicubic', sharpen)
  x = frames.permute(0, 3, 1, 2).float()
  if x.shape[-2:] != (target_h, target_w):
    x = F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=False) if mode != 'lanczos' else _lanczos(x, target_h, target_w)
  if sharpen:
    x = x + sharpen * (x - _gaussian_blur(x))
  return x.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1)


def _stretch_index(n_out, n_in):
  """For each of n_in source positions, the stored sample it should copy.

  The networks bilinearly downsample (align_corners=False), so output j reads source
  positions floor(c_j) and floor(c_j)+1 with c_j = (j+0.5)*n_in/n_out - 0.5. Giving both
  of them sample j makes that downsample return the stored frame exactly. Those pairs never
  overlap (n_in/n_out > 2); the remaining positions are never read and copy their nearest sample.
  """
  j = torch.arange(n_out, dtype=torch.float64)
  lo = ((j + 0.5) * n_in / n_out - 0.5).floor().long().clamp(0, n_in - 2)
  s = torch.arange(n_in, dtype=torch.float64)
  idx = ((s + 0.5) * n_out / n_in - 0.5).round().long().clamp(0, n_out - 1)
  idx[lo], idx[lo + 1] = torch.arange(n_out), torch.arange(n_out)
  return idx


def _stretch(frames, out_h, out_w):
  rows = _stretch_index(frames.shape[1], out_h).to(frames.device)
  cols = _stretch_index(frames.shape[2], out_w).to(frames.device)
  return frames[:, rows][:, :, cols]


def _lanczos_weights(in_size, out_size, a, device):
  scale = in_size / out_size
  centers = (torch.arange(out_size, device=device, dtype=torch.float32) + 0.5) * scale - 0.5
  taps = torch.arange(-a + 1, a + 1, device=device, dtype=torch.float32)
  idx = centers.floor()[:, None] + taps[None, :]
  d = centers[:, None] - idx
  w = torch.sinc(d) * torch.sinc(d / a)
  w = w / w.sum(dim=1, keepdim=True)
  W = torch.zeros(out_size, in_size, device=device)
  W.scatter_add_(1, idx.long().clamp(0, in_size - 1), w)
  return W


def _lanczos(x, out_h, out_w, a=3):
  Wh = _lanczos_weights(x.shape[-2], out_h, a, x.device)
  Ww = _lanczos_weights(x.shape[-1], out_w, a, x.device)
  return torch.einsum('oh,bchw,pw->bcop', Wh, x, Ww)


def decode_batches(video_path, batch_size=16, device=None, mode=UPSCALE_MODE, sharpen=SHARPEN):
  """Yields (B, H, W, 3) uint8 frames at camera resolution on `device`."""
  device = device or pick_device()
  container = av.open(video_path)
  buf = []
  for frame in container.decode(container.streams.video[0]):
    buf.append(yuv420_to_rgb(frame))
    if len(buf) == batch_size:
      yield upscale(torch.stack(buf).to(device), mode, sharpen)
      buf = []
  if buf:
    yield upscale(torch.stack(buf).to(device), mode, sharpen)
  container.close()


if __name__ == "__main__":
  import sys
  src, dst = sys.argv[1], sys.argv[2]
  n = 0
  with open(dst, 'wb') as f:
    for batch in decode_batches(src):
      f.write(batch.cpu().contiguous().numpy().tobytes())
      n += batch.shape[0]
  print(f"saved {n} frames")
