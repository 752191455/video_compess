#!/usr/bin/env python
"""Writes the video as raw rgb24 frames at 512x384 to stdout, downsampled exactly the way
SegNet/PoseNet preprocess their input (bilinear, align_corners=False), so the encoder sees
the same pixels the networks do.

  python -m submissions.svtav1_fg22_crf36_p0.downscale videos/0.mkv | ffmpeg -f rawvideo -pix_fmt rgb24 -s 512x384 -i - ...
"""
import sys, av, torch
import torch.nn.functional as F
from frame_utils import yuv420_to_rgb, segnet_model_input_size


def main(src):
  w, h = segnet_model_input_size
  container = av.open(src)
  out = sys.stdout.buffer
  for frame in container.decode(container.streams.video[0]):
    x = yuv420_to_rgb(frame).permute(2, 0, 1)[None].float()
    x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
    out.write(x[0].clamp(0, 255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy().tobytes())
  container.close()


if __name__ == '__main__':
  main(sys.argv[1])
