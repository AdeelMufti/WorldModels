from datetime import datetime, timezone
import os
from scipy.misc import imsave
import numpy as np


def pre_process_image_tensor(images):
    if images.dtype != np.float32:
        images = images.astype(np.float32) / 255.
    if images.shape[-1] == 3:
        images = np.rollaxis(images, 3, 1)
    return images


def post_process_image_tensor(images):
    if images.dtype != np.uint8:
        images = (images * 255).astype('uint8')
    if images.shape[-1] != 3:
        images = np.rollaxis(images, 1, 4)
    return images


def save_images_collage(images, save_path, pre_processed=True):
    if pre_processed:
        images = post_process_image_tensor(images)

    npad = ((0, 0), (2, 2), (2, 2), (0, 0))
    images = np.pad(images, pad_width=npad, mode='constant', constant_values=255)

    n_samples = images.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples // rows

    if images.ndim == 2:
        images = np.reshape(images, (images.shape[0], int(np.sqrt(images.shape[1])), int(np.sqrt(images.shape[1]))))

    if images.ndim == 4:
        h, w = images[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif images.ndim == 3:
        h, w = images[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, images in enumerate(images):
        j = n // nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = images

    imsave(save_path, img)


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def log(id, message):
    print(str(datetime.now(timezone.utc)) + " [" + str(id) + "] " + str(message))
