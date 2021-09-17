import random
import numpy as np
import cv2

from PIL import Image
from skimage import transform
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torch.utils.data.dataloader import default_collate


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def voice_pipeline(info, test_mode):
    path = info['path']
    image = Image.open(path).convert('RGB')
    image = np.array(image)
    # crop
    src_landmark = info.get('src_landmark')
    tgz_landmark = info.get('tgz_landmark')
    crop_size = info.get('crop_size')
    if not (src_landmark is None or tgz_landmark is None or crop_size is None):
        tform = transform.SimilarityTransform()
        tform.estimate(tgz_landmark, src_landmark)
        M = tform.params[0:2, :]
        image = cv2.warpAffine(image, M, crop_size, borderValue=0.0)
    # normalize to [-1, 1]
    image = ((image - 127.5) / 127.5)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    if not test_mode and random.random() > 0.5:
        image = np.flip(image, axis=2).copy()
    return image


def image_pipeline(info, test_mode):
    path = info['path']
    image = Image.open(path).convert('RGB')
    image = np.array(image)
    # crop
    src_landmark = info.get('src_landmark')
    tgz_landmark = info.get('tgz_landmark')
    crop_size = info.get('crop_size')
    if not (src_landmark is None or tgz_landmark is None or crop_size is None):
        tform = transform.SimilarityTransform()
        tform.estimate(tgz_landmark, src_landmark)
        M = tform.params[0:2, :]
        image = cv2.warpAffine(image, M, crop_size, borderValue=0.0)
    # normalize to [-1, 1]
    image = ((image - 127.5) / 127.5)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    if not test_mode and random.random() > 0.5:
        image = np.flip(image, axis=2).copy()
    return image


def get_collate_fn(duration, sample_rate):
    # collate_fn
    duration[0] < duration[1]
    min_num_frames = duration[0] * sample_rate
    max_num_frames = duration[1] * sample_rate
    def collate_fn(batch):
        assert min_num_frames <= max_num_frames
        num_crop_frame = np.random.randint(
                min_num_frames, max_num_frames + 1)
        pt = np.random.randint(0, max_num_frames - num_crop_frame + 1)
        batch = [(item[0], item[1][..., pt:pt+num_crop_frame], item[2]) for item in batch]
        return default_collate(batch)
    return collate_fn

def get_metrics(scores, labels):
    # eer and auc
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    eer = 100. * brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    auc = 100. * metrics.auc(fpr, tpr)

    # get acc
    tnr = 1. - fpr
    pos_num = labels.count(1)
    neg_num = labels.count(0)
    acc = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)

    return acc, eer, auc
