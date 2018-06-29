import gzip
import numpy as np
import os
import random
import gc
from multiprocessing import cpu_count, Pool

from chainer import dataset
import chainer.functions as F

from lib.utils import log, pre_process_image_tensor


def load_frames_worker(frames_file):
    with gzip.GzipFile(frames_file, "r") as file:
        rollout_frames = pre_process_image_tensor(np.load(file))
    return rollout_frames


def load_model_npz_worker(files):
    npz1, npz2 = files

    npz = np.load(npz1)
    mu = npz['mu']
    ln_var = npz['ln_var']
    npz.close()

    npz = np.load(npz2)
    action = npz['action']
    npz.close()

    return mu, ln_var, action


class VisionDataset(dataset.DatasetMixin):
    def __init__(self, dir='', load_batch_size=10, shuffle=True, verbose=True):
        rollouts = os.listdir(dir)
        rollouts_counts = {}

        for rollout in rollouts:
            count_file = os.path.join(dir, rollout, "count")
            if os.path.exists(count_file):
                with open(count_file, 'r') as count_file:
                    count = int(count_file.read())
                    rollouts_counts[rollout] = count

        rollouts = list(rollouts_counts.keys())

        if shuffle:
            random.shuffle(rollouts)
        else:
            rollouts = sorted(rollouts, key=lambda x: int(x))

        total_batches = len(rollouts) // load_batch_size
        if len(rollouts) % load_batch_size != 0:
            total_batches += 1

        self.batch = -1
        self.total_batches = total_batches
        self.dir = dir
        self.shuffle = shuffle
        self.verbose = verbose
        self.load_batch_size = load_batch_size
        self.rollouts = rollouts
        self.total_count = sum(rollouts_counts.values())
        self.rollouts_counts = rollouts_counts

        self.reset_indices()

        self.load_batch(0)

    def reset_indices(self):
        if self.verbose:
            log("VisionDataset", "*** Creating list of indices")
        running_count = 0
        absolute_indices = []
        for batch in range(self.total_batches):
            batch_start_idx = batch * self.load_batch_size
            batch_end_idx = (batch + 1) * self.load_batch_size
            batch_rollouts = self.rollouts[batch_start_idx:batch_end_idx]
            count = sum([self.rollouts_counts[rollout] for rollout in batch_rollouts])
            absolute_start_idx = running_count
            absolute_end_idx = running_count + count
            running_count = running_count + count
            if self.verbose:
                log("VisionDataset", "*** Batch " + str(batch) + ", from index " + str(batch_start_idx) + ":" + str(
                    batch_end_idx) + ", of rollouts " + str(batch_rollouts) + ", with " + str(
                    count) + " frames in this batch, goes from absolute index " + str(absolute_start_idx) + ":" + str(
                    absolute_end_idx))
            absolute_indices.append([absolute_start_idx, absolute_end_idx])
        self.absolute_indices = absolute_indices

    def load_batch(self, batch):
        if self.batch == batch:
            return

        if batch == 0 and self.batch > 0 and self.shuffle:
            random.shuffle(self.rollouts)
            self.reset_indices()

        self.batch_frames = None
        gc.collect()

        if self.verbose:
            log("VisionDataset", "*** Loading batch " + str(batch))

        batch_start_idx = batch * self.load_batch_size
        batch_end_idx = (batch + 1) * self.load_batch_size
        batch_rollouts = self.rollouts[batch_start_idx:batch_end_idx]
        batch_frames = None
        batch_rollouts_counts = {}
        frames_files = []
        for rollout in batch_rollouts:
            batch_rollouts_counts[rollout] = self.rollouts_counts[rollout]
            frames_files.append(os.path.join(self.dir, rollout, "frames.npy.gz"))
        pool = Pool(cpu_count())
        all_rollout_frames = pool.map(load_frames_worker, frames_files)
        pool.close()
        pool.join()
        batch_frames = np.concatenate(all_rollout_frames)
        if self.shuffle:
            shuffled_indices = np.random.permutation(np.arange(batch_frames.shape[0]))
            batch_frames = batch_frames[shuffled_indices]
        if self.verbose:
            log("VisionDataset", "*** Loaded batch " + str(batch) + ", from index " + str(batch_start_idx) + ":" + str(
                batch_end_idx) + ", of rollouts " + str(batch_rollouts) + ", with " + str(
                batch_frames.shape[0]) + " total frames in this batch. Each rollout has count: " + str(
                batch_rollouts_counts))
        self.batch_rollouts = batch_rollouts
        self.batch_rollouts_counts = batch_rollouts_counts
        self.batch_frames = batch_frames
        self.batch = batch

    def get_current_batch_size(self):
        return self.batch_frames.shape[0]

    def get_total_batches(self):
        return self.total_batches

    def get_current_batch(self):
        return self.batch_frames, self.batch_rollouts, self.batch_rollouts_counts

    def __len__(self):
        return self.total_count

    def get_example(self, i):
        absolute_start_idx = self.absolute_indices[self.batch][0]
        absolute_end_idx = self.absolute_indices[self.batch][1]
        if i < absolute_start_idx or i >= absolute_end_idx:
            for batch, absolute_indices in enumerate(self.absolute_indices):
                absolute_start_idx = absolute_indices[0]
                absolute_end_idx = absolute_indices[1]
                if i >= absolute_start_idx and i < absolute_end_idx:
                    self.load_batch(batch)
                    break
        return self.batch_frames[i - absolute_start_idx]


class ModelDataset(dataset.DatasetMixin):
    def __init__(self, dir='', load_batch_size=10, verbose=True):
        rollouts = os.listdir(dir)
        rollouts_counts = {}

        for rollout in rollouts:
            count_file = os.path.join(dir, rollout, "count")
            if os.path.exists(count_file):
                with open(count_file, 'r') as count_file:
                    count = int(count_file.read())
                    rollouts_counts[rollout] = count - 1  # -1 b/c last frame doesn't have a next frame

        rollouts = list(rollouts_counts.keys())

        # Sort by the longest rollouts up front, for chainer's LSTM, at least for the first epoch
        rollouts = sorted(rollouts, key=lambda x: -rollouts_counts[x])

        total_batches = len(rollouts) // load_batch_size
        if len(rollouts) % load_batch_size != 0:
            total_batches += 1

        for batch in range(total_batches):
            batch_start_idx = batch * load_batch_size
            batch_end_idx = (batch + 1) * load_batch_size
            batch_rollouts = rollouts[batch_start_idx:batch_end_idx]
            if verbose:
                log("ModelDataset", "*** Batch " + str(batch) + ", from index " + str(batch_start_idx) + ":" + str(
                    batch_end_idx) + ", will be of rollouts " + str(batch_rollouts))

        self.batch = -1
        self.last_index = -1
        self.total_batches = total_batches
        self.dir = dir
        self.verbose = verbose
        self.load_batch_size = load_batch_size
        self.rollouts = rollouts
        self.rollouts_counts = rollouts_counts

        self.load_batch(0)

    def load_batch(self, batch):
        if self.batch == batch:
            return

        if batch == 0 and self.batch > 0:
            random.shuffle(self.rollouts)

        self.z_t = None
        self.z_t_plus_1 = None
        self.action = None
        gc.collect()

        if self.verbose:
            log("ModelDataset", "*** Loading batch " + str(batch))

        batch_start_idx = batch * self.load_batch_size
        batch_end_idx = (batch + 1) * self.load_batch_size
        batch_rollouts = self.rollouts[batch_start_idx:batch_end_idx]
        batch_rollouts_counts = {}
        files = []
        for rollout in batch_rollouts:
            batch_rollouts_counts[rollout] = self.rollouts_counts[rollout]
            files.append(
                (os.path.join(self.dir, rollout, "mu+ln_var.npz"),
                 os.path.join(self.dir, rollout, "misc.npz")))
        pool = Pool(cpu_count())
        data = pool.map(load_model_npz_worker, files)
        pool.close()
        pool.join()
        mu = []
        ln_var = []
        action = []
        for rollout_mu, rollout_ln_var, rollout_action in data:
            mu.append(rollout_mu)
            ln_var.append(rollout_ln_var)
            action.append(rollout_action)

        if self.verbose:
            log("ModelDataset", "*** Loaded batch " + str(batch) + ", from index " + str(batch_start_idx) + ":" + str(
                batch_end_idx) + ", of rollouts " + str(batch_rollouts) + ", with " + str(
                len(mu)) + " total rollouts in this batch. Each rollout has count: " + str(
                batch_rollouts_counts))

        self.batch_rollouts = batch_rollouts
        self.batch_rollouts_counts = batch_rollouts_counts
        self.mu = mu
        self.ln_var = ln_var
        self.action = action
        self.batch = batch

    def get_current_batch_size(self):
        return len(self.batch_rollouts)

    def __len__(self):
        return len(self.rollouts)

    def get_example(self, i):
        batch = i // self.load_batch_size
        self.load_batch(batch)
        index = i % self.load_batch_size

        # In case we have all rollouts loaded in memory, and
        # are not doing batched loading, shuffle every epoch:
        if self.load_batch_size >= len(self.rollouts):
            if index == 0 and self.last_index > 0:
                shuffled = list(zip(self.mu, self.ln_var, self.action))
                random.shuffle(shuffled)
                self.mu, self.ln_var, self.action = zip(*shuffled)
            self.last_index = index

        # reconstruct every time, prevent overfitting:
        z_t = F.gaussian(self.mu[index], self.ln_var[index]).data
        z_t_plus_1 = z_t[1:]
        z_t = z_t[0:z_t.shape[0] - 1]

        done = np.zeros((z_t_plus_1.shape[0], 1)).astype(np.int32)
        done[-1, 0] = 1.

        return z_t, z_t_plus_1, self.action[index], done
