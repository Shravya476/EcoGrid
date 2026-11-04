"""Turns satellite image crops into simple numbers. Same recipe that was used when the cloud models were trained."""
import numpy as np


def frame_features(img):
    centre = img[24:40, 24:40]                         # the 16 x 16 square right over the plant
    return [np.nanmean(img), np.nanstd(img), np.nanmin(img),
            np.nanpercentile(img, 10), np.nanpercentile(img, 90),
            np.nanmean(img < 280), np.nanmean(img < 265),
            np.nanmean(centre), np.nanmean(centre < 280)]


def example_features(frames, now, hour, month):
    """frames: 3 images (hours h-2, h-1, h), each 64 x 64; now: sun ratio at hour h."""
    f = [frame_features(frames[t]) for t in range(3)]
    now_f, prev_f, old_f = np.array(f[2]), np.array(f[1]), np.array(f[0])
    return np.concatenate([now_f, now_f - old_f, now_f - prev_f, [now, hour, month]])


def estimate_features(frame, hour, month):
    """frame: the newest 64 x 64 image. Used to ESTIMATE the current sunshine."""
    return np.concatenate([frame_features(frame), [hour, month]])
