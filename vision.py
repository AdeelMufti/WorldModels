import argparse
import os
import re
import gc

import chainer.functions as F
import chainer.links as L
import chainer
from chainer import training
from chainer.training import extensions
try:
    import cupy as cp
except Exception as e:
    None

import numpy as np

from lib.data import VisionDataset
from lib.utils import save_images_collage, mkdir, log, pre_process_image_tensor, post_process_image_tensor

ID = "vision"


class CVAE(chainer.Chain):
    def __init__(self, n_latent):
        self.n_latent = n_latent
        super(CVAE, self).__init__(
            e_c0=L.Convolution2D(None, 32, 4, 2),
            e_c1=L.Convolution2D(None, 64, 4, 2),
            e_c2=L.Convolution2D(None, 128, 4, 2),
            e_c3=L.Convolution2D(None, 256, 4, 2),
            e_mu=L.Linear(None, n_latent),
            e_ln_var=L.Linear(None, n_latent),

            d_l0=L.Linear(n_latent, 1024),
            d_dc0=L.Deconvolution2D(None, 128, 5, 2),
            d_dc1=L.Deconvolution2D(None, 64, 5, 2),
            d_dc2=L.Deconvolution2D(None, 32, 6, 2),
            d_dc3=L.Deconvolution2D(None, 3, 6, 2),
        )

    def __call__(self, frames, pre_process=False):
        if len(frames.shape) == 3:
            frames = F.expand_dims(frames, 0)
        if pre_process:
            frames = pre_process_image_tensor(frames)
        frames_variational = self.decode(self.encode(frames, return_z=True))
        if pre_process:
            frames_variational = post_process_image_tensor(frames_variational)
        return frames_variational

    def encode(self, frames, return_z=False):
        if len(frames.shape) == 3:
            frames = F.expand_dims(frames, 0)
        h = F.relu(self.e_c0(frames))
        h = F.relu(self.e_c1(h))
        h = F.relu(self.e_c2(h))
        h = F.relu(self.e_c3(h))
        h = F.reshape(h, (-1, 1024))
        mu = self.e_mu(h)
        ln_var = self.e_ln_var(h)
        if return_z:
            return F.gaussian(mu, ln_var)
        else:
            return mu, ln_var

    def decode(self, z):
        if len(z.shape) == 1:
            z = F.expand_dims(z, 0)
        h = self.d_l0(z)
        h = F.reshape(h, (-1, 1024, 1, 1))
        h = F.relu(self.d_dc0(h))
        h = F.relu(self.d_dc1(h))
        h = F.relu(self.d_dc2(h))
        h = F.sigmoid(self.d_dc3(h))
        return h

    def get_loss_func(self, kl_tolerance=0.5):
        self.kl_tolerance = kl_tolerance
        def lf(frames):
            mu, ln_var = self.encode(frames)
            z = F.gaussian(mu, ln_var)
            frames_flat = F.reshape(frames, (-1, frames.shape[1] * frames.shape[2] * frames.shape[3]))
            variational_flat = F.reshape(self.decode(z), (-1, frames.shape[1] * frames.shape[2] * frames.shape[3]))
            rec_loss = F.sum(F.square(frames_flat - variational_flat), axis=1)  # l2 reconstruction loss
            rec_loss = F.mean(rec_loss)
            kl_loss = F.sum(F.gaussian_kl_divergence(mu, ln_var, reduce="no"), axis=1)
            if self._cpu:
                kl_tolerance = np.asarray(self.kl_tolerance * self.n_latent).astype(np.float32)
            else:
                kl_tolerance = cp.asarray(self.kl_tolerance * self.n_latent).astype(cp.float32)
            kl_loss = F.maximum(kl_loss, F.broadcast_to(kl_tolerance, kl_loss.shape))
            kl_loss = F.mean(kl_loss)
            loss = rec_loss + kl_loss
            chainer.report({'loss': loss}, observer=self)
            chainer.report({'kl_loss': kl_loss}, observer=self)
            chainer.report({'rec_loss': rec_loss}, observer=self)
            return loss
        return lf


class Sampler(chainer.training.Extension):
    def __init__(self, model, args, output_dir, frames, z):
        self.model = model
        self.args = args
        self.output_dir = output_dir
        self.frames = frames
        self.z = z

    def __call__(self, trainer):
        if self.args.gpu >= 0:
            self.model.to_cpu()

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            frames_variational = self.model(self.frames)
            save_images_collage(frames_variational.data,
                                os.path.join(self.output_dir,
                                             'train_reconstructed_{}.png'.format(trainer.updater.iteration)))

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            frames_variational = self.model.decode(self.z)
            save_images_collage(frames_variational.data,
                                os.path.join(self.output_dir, 'sampled_{}.png'.format(trainer.updater.iteration)))

        if self.args.gpu >= 0:
            self.model.to_gpu()


def main():
    parser = argparse.ArgumentParser(description='World Models ' + ID)
    parser.add_argument('--data_dir', '-d', default="/data/wm", help='The base data/output directory')
    parser.add_argument('--game', default='CarRacing-v0',
                        help='Game to use')  # https://gym.openai.com/envs/CarRacing-v0/
    parser.add_argument('--experiment_name', default='experiment_1', help='To isolate its files from others')
    parser.add_argument('--load_batch_size', default=10, type=int,
                        help='Load game frames in batches so as not to run out of memory')
    parser.add_argument('--model', '-m', default='',
                        help='Initialize the model from given file, or "default" for one in data folder')
    parser.add_argument('--no_resume', action='store_true', help='Don''t auto resume from the latest snapshot')
    parser.add_argument('--resume_from', '-r', default='', help='Resume the optimization from a specific snapshot')
    parser.add_argument('--test', action='store_true', help='Generate samples only')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=1, type=int, help='number of epochs to learn')
    parser.add_argument('--snapshot_interval', '-s', default=100, type=int,
                        help='100 = snapshot every 100itr*batch_size imgs processed')
    parser.add_argument('--z_dim', '-z', default=32, type=int, help='dimension of encoded vector')
    parser.add_argument('--batch_size', '-b', type=int, default=100, help='learning minibatch size')
    parser.add_argument('--no_progress_bar', '-p', action='store_true', help='Display progress bar during training')
    parser.add_argument('--kl_tolerance', type=float, default=0.5, help='')

    args = parser.parse_args()
    log(ID, "args =\n " + str(vars(args)).replace(",", ",\n "))

    output_dir = os.path.join(args.data_dir, args.game, args.experiment_name, ID)
    random_rollouts_dir = os.path.join(args.data_dir, args.game, args.experiment_name, 'random_rollouts')
    mkdir(output_dir)

    max_iter = 0
    auto_resume_file = None
    files = os.listdir(output_dir)
    for file in files:
        if re.match(r'^snapshot_iter_', file):
            iter = int(re.search(r'\d+', file).group())
            if (iter > max_iter):
                max_iter = iter
    if max_iter > 0:
        auto_resume_file = os.path.join(output_dir, "snapshot_iter_{}".format(max_iter))

    model = CVAE(args.z_dim)

    if args.model:
        if args.model == 'default':
            args.model = os.path.join(output_dir, ID + ".model")
        log(ID, "Loading saved model from: " + args.model)
        chainer.serializers.load_npz(args.model, model)

    optimizer = chainer.optimizers.Adam(alpha=0.0001)
    optimizer.setup(model)

    log(ID, "Loading training data")
    train = VisionDataset(dir=random_rollouts_dir, load_batch_size=args.load_batch_size, shuffle=True, verbose=True)
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size, shuffle=False)

    updater = training.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu, loss_func=model.get_loss_func(args.kl_tolerance))

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output_dir)
    trainer.extend(extensions.snapshot(), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(100 if args.gpu >= 0 else 10, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/kl_loss', 'main/rec_loss', 'elapsed_time']))
    if not args.no_progress_bar:
        trainer.extend(extensions.ProgressBar(update_interval=100 if args.gpu >= 0 else 10))

    sample_idx = np.random.choice(range(train.get_current_batch_size()), 64, replace=False)
    sample_frames = chainer.Variable(np.asarray(train[sample_idx]))
    np.random.seed(31337)
    sample_z = chainer.Variable(np.random.normal(0, 1, (64, args.z_dim)).astype(np.float32))
    save_images_collage(sample_frames.data, os.path.join(output_dir, 'train.png'))
    sampler = Sampler(model, args, output_dir, sample_frames, sample_z)
    trainer.extend(sampler, trigger=(args.snapshot_interval, 'iteration'))

    if args.resume_from:
        log(ID, "Resuming trainer manually from snapshot: " + args.resume_from)
        chainer.serializers.load_npz(args.resume_from, trainer)
    elif not args.no_resume and auto_resume_file is not None:
        log(ID, "Auto resuming trainer from last snapshot: " + auto_resume_file)
        chainer.serializers.load_npz(auto_resume_file, trainer)

    if not args.test:
        log(ID, "Starting training")
        trainer.run()
        log(ID, "Done training")
        log(ID, "Saving model")
        chainer.serializers.save_npz(os.path.join(output_dir, ID + ".model"), model)

    if args.test:
        log(ID, "Saving test samples")
        sampler(trainer)

    if not args.test:
        log(ID, "Saving latent z's for all training data")
        train = VisionDataset(dir=random_rollouts_dir, load_batch_size=args.load_batch_size, shuffle=False,
                              verbose=True)
        total_batches = train.get_total_batches()
        for batch in range(total_batches):
            gc.collect()
            train.load_batch(batch)
            batch_frames, batch_rollouts, batch_rollouts_counts = train.get_current_batch()
            mu = None
            ln_var = None
            splits = batch_frames.shape[0] // args.batch_size
            if batch_frames.shape[0] % args.batch_size != 0:
                splits += 1
            for i in range(splits):
                start_idx = i * args.batch_size
                end_idx = (i + 1) * args.batch_size
                sample_frames = batch_frames[start_idx:end_idx]
                if args.gpu >= 0:
                    sample_frames = chainer.Variable(cp.asarray(sample_frames))
                else:
                    sample_frames = chainer.Variable(sample_frames)
                this_mu, this_ln_var = model.encode(sample_frames)
                this_mu = this_mu.data
                this_ln_var = this_ln_var.data
                if args.gpu >= 0:
                    this_mu = cp.asnumpy(this_mu)
                    this_ln_var = cp.asnumpy(this_ln_var)
                if mu is None:
                    mu = this_mu
                    ln_var = this_ln_var
                else:
                    mu = np.concatenate((mu, this_mu), axis=0)
                    ln_var = np.concatenate((ln_var, this_ln_var), axis=0)
            running_count = 0
            for rollout in batch_rollouts:
                rollout_dir = os.path.join(random_rollouts_dir, rollout)
                rollout_count = batch_rollouts_counts[rollout]
                start_idx = running_count
                end_idx = running_count + rollout_count
                this_mu = mu[start_idx:end_idx]
                this_ln_var = ln_var[start_idx:end_idx]
                np.savez_compressed(os.path.join(rollout_dir, "mu+ln_var.npz"), mu=this_mu, ln_var=this_ln_var)
                running_count = running_count + rollout_count
            log(ID, "> Processed z's for rollouts " + str(batch_rollouts))
            # Free up memory:
            batch_frames = None
            mu = None
            ln_var = None

    log(ID, "Done")


if __name__ == '__main__':
    main()
