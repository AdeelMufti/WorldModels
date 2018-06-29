import argparse
import os
import re
import math

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
try:
    import cupy as cp
except Exception as e:
    None

import numpy as np
import imageio
import numba

from lib.utils import log, mkdir, save_images_collage, post_process_image_tensor
from lib.data import ModelDataset
from vision import CVAE

ID = "model"


@numba.jit(nopython=True)
def optimized_sampling(output_dim, temperature, coef, mu, ln_var):
    mus = np.zeros(output_dim)
    ln_vars = np.zeros(output_dim)
    for i in range(output_dim):
        cumulative_probability = 0.
        r = np.random.uniform(0., 1.)
        index = len(coef)-1
        for j, probability in enumerate(coef[i]):
            cumulative_probability = cumulative_probability + probability
            if r <= cumulative_probability:
                index = j
                break
        for j, this_mu in enumerate(mu[i]):
            if j == index:
                mus[i] = this_mu
                break
        for j, this_ln_var in enumerate(ln_var[i]):
            if j == index:
                ln_vars[i] = this_ln_var
                break
    z_t_plus_1 = mus + np.exp(ln_vars) * np.random.randn(output_dim) * np.sqrt(temperature)
    return z_t_plus_1


class MDN_RNN(chainer.Chain):
    def __init__(self, hidden_dim=256, output_dim=32, k=5, predict_done=False):
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.predict_done = predict_done
        init_dict = {
            "rnn_layer": L.LSTM(None, hidden_dim),
            "coef_layer": L.Linear(None, k * output_dim),
            "mu_layer": L.Linear(None, k * output_dim),
            "ln_var_layer": L.Linear(None, k * output_dim)
        }
        if predict_done:
            init_dict["done_layer"] = L.Linear(None, 1)
        super(MDN_RNN, self).__init__(**init_dict)

    def __call__(self, z_t, action, temperature=1.0):
        k = self.k
        output_dim = self.output_dim

        if len(z_t.shape) == 1:
            z_t = F.expand_dims(z_t, 0)
        if len(action.shape) == 1:
            action = F.expand_dims(action, 0)

        output = self.fprop(F.concat((z_t, action)))
        if self.predict_done:
            coef, mu, ln_var, done = output
        else:
            coef, mu, ln_var = output

        coef = F.reshape(coef, (-1, k))
        mu = F.reshape(mu, (-1, k))
        ln_var = F.reshape(ln_var, (-1, k))

        coef /= temperature
        coef = F.softmax(coef,axis=1)

        if self._cpu:
            z_t_plus_1 = optimized_sampling(output_dim, temperature, coef.data, mu.data, ln_var.data).astype(np.float32)
        else:
            coef = cp.asnumpy(coef.data)
            mu = cp.asnumpy(mu.data)
            ln_var = cp.asnumpy(ln_var.data)
            z_t_plus_1 = optimized_sampling(output_dim, temperature, coef, mu, ln_var).astype(np.float32)
            z_t_plus_1 = chainer.Variable(cp.asarray(z_t_plus_1))

        if self.predict_done:
            return z_t_plus_1, F.sigmoid(done)
        else:
            return z_t_plus_1

    def fprop(self, input):
        h = self.rnn_layer(input)
        coef = self.coef_layer(h)
        mu = self.mu_layer(h)
        ln_var = self.ln_var_layer(h)

        if self.predict_done:
            done = self.done_layer(h)

        if self.predict_done:
            return coef, mu, ln_var, done
        else:
            return coef, mu, ln_var

    def get_loss_func(self):
        def lf(z_t, z_t_plus_1, action, done_label, reset=True):
            k = self.k
            output_dim = self.output_dim
            if reset:
                self.reset_state()

            output = self.fprop(F.concat((z_t, action)))
            if self.predict_done:
                coef, mu, ln_var, done = output
            else:
                coef, mu, ln_var = output

            coef = F.reshape(coef, (-1, output_dim, k))
            coef = F.softmax(coef, axis=2)
            mu = F.reshape(mu, (-1, output_dim, k))
            ln_var = F.reshape(ln_var, (-1, output_dim, k))

            z_t_plus_1 = F.repeat(z_t_plus_1, k, 1).reshape(-1, output_dim, k)

            normals = F.sum(
                coef * F.exp(-F.gaussian_nll(z_t_plus_1, mu, ln_var, reduce='no'))
                ,axis=2)
            densities = F.sum(normals, axis=1)
            nll = -F.log(densities)

            loss = F.sum(nll)

            if self.predict_done:
                done_loss = F.sigmoid_cross_entropy(done.reshape(-1,1), done_label, reduce="no")
                done_loss *= (1. + done_label.astype("float32")*9.)
                done_loss = F.mean(done_loss)
                loss = loss + done_loss

            return loss
        return lf

    def reset_state(self):
        self.rnn_layer.reset_state()

    def get_h(self):
        return self.rnn_layer.h

    def get_c(self):
        return self.rnn_layer.c


class ImageSampler(chainer.training.Extension):
    def __init__(self, model, vision, args, output_dir, z_t, action):
        self.model = model
        self.vision = vision
        self.args = args
        self.output_dir = output_dir
        self.z_t = z_t
        self.action = action

    def __call__(self, trainer):
        if self.args.gpu >= 0:
            self.model.to_cpu()
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            self.model.reset_state()
            z_t_plus_1s = []
            dones = []
            for i in range(self.z_t.shape[0]):
                output = self.model(self.z_t[i], self.action[i], temperature=self.args.sample_temperature)
                if self.args.predict_done:
                    z_t_plus_1, done = output
                    z_t_plus_1 = z_t_plus_1.data
                    done = done.data
                else:
                    z_t_plus_1 = output.data
                z_t_plus_1s.append(z_t_plus_1)
                if self.args.predict_done:
                    dones.append(done[0])
            z_t_plus_1s = np.asarray(z_t_plus_1s)
            dones = np.asarray(dones).reshape(-1)
            img_t_plus_1 = post_process_image_tensor(self.vision.decode(z_t_plus_1s).data)
            if self.args.predict_done:
                img_t_plus_1[np.where(dones >= 0.5), :, :, :] = 0  # Make all the done's black
            save_images_collage(img_t_plus_1,
                                os.path.join(self.output_dir,
                                             'train_t_plus_1_{}.png'.format(trainer.updater.iteration)),
                                pre_processed=False)
        if self.args.gpu >= 0:
            self.model.to_gpu()


class TBPTTUpdater(training.updaters.StandardUpdater):
    def __init__(self, train_iter, optimizer, device, loss_func, sequence_length):
        self.sequence_length = sequence_length
        super(TBPTTUpdater, self).__init__(
            train_iter, optimizer, device=device,
            loss_func=loss_func)

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()
        total_loss = 0
        z_t, z_t_plus_1, action, done = self.converter(batch, self.device)
        z_t = chainer.Variable(z_t[0])
        z_t_plus_1 = chainer.Variable(z_t_plus_1[0])
        action = chainer.Variable(action[0])
        done = chainer.Variable(done[0])
        for i in range(math.ceil(z_t.shape[0]/self.sequence_length)):
            start_idx = i*self.sequence_length
            end_idx = (i+1)*self.sequence_length
            loss = self.loss_func(z_t[start_idx:end_idx].data,
                                  z_t_plus_1[start_idx:end_idx].data,
                                  action[start_idx:end_idx].data,
                                  done[start_idx:end_idx].data,
                                  True if i==0 else False)
            optimizer.target.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            total_loss += loss

        chainer.report({'loss': total_loss})


def main():
    parser = argparse.ArgumentParser(description='World Models ' + ID)
    parser.add_argument('--data_dir', '-d', default="/data/wm", help='The base data/output directory')
    parser.add_argument('--game', default='CarRacing-v0',
                        help='Game to use')  # https://gym.openai.com/envs/CarRacing-v0/
    parser.add_argument('--experiment_name', default='experiment_1', help='To isolate its files from others')
    parser.add_argument('--load_batch_size', default=100, type=int,
                        help='Load rollouts in batches so as not to run out of memory')
    parser.add_argument('--model', '-m', default='',
                        help='Initialize the model from given file, or "default" for one in data folder')
    parser.add_argument('--no_resume', action='store_true', help='Don''t auto resume from the latest snapshot')
    parser.add_argument('--resume_from', '-r', default='', help='Resume the optimization from a specific snapshot')
    parser.add_argument('--test', action='store_true', help='Generate samples only')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=20, type=int, help='number of epochs to learn')
    parser.add_argument('--snapshot_interval', '-s', default=200, type=int, help='snapshot every x games')
    parser.add_argument('--z_dim', '-z', default=32, type=int, help='dimension of encoded vector')
    parser.add_argument('--hidden_dim', default=256, type=int, help='LSTM hidden units')
    parser.add_argument('--mixtures', default=5, type=int, help='number of gaussian mixtures for MDN')
    parser.add_argument('--no_progress_bar', '-p', action='store_true', help='Display progress bar during training')
    parser.add_argument('--predict_done', action='store_true', help='Whether MDN-RNN should also predict done state')
    parser.add_argument('--sample_temperature', default=1., type=float, help='Temperature for generating samples')
    parser.add_argument('--gradient_clip', default=0., type=float, help='Clip grads L2 norm threshold. 0 = no clip')
    parser.add_argument('--sequence_length', type=int, default=128, help='sequence length for LSTM for TBPTT')

    args = parser.parse_args()
    log(ID, "args =\n " + str(vars(args)).replace(",", ",\n "))

    output_dir = os.path.join(args.data_dir, args.game, args.experiment_name, ID)
    mkdir(output_dir)
    random_rollouts_dir = os.path.join(args.data_dir, args.game, args.experiment_name, 'random_rollouts')
    vision_dir = os.path.join(args.data_dir, args.game, args.experiment_name, 'vision')

    log(ID, "Starting")

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

    model = MDN_RNN(args.hidden_dim, args.z_dim, args.mixtures, args.predict_done)
    vision = CVAE(args.z_dim)
    chainer.serializers.load_npz(os.path.join(vision_dir, "vision.model"), vision)

    if args.model:
        if args.model == 'default':
            args.model = os.path.join(output_dir, ID + ".model")
        log(ID, "Loading saved model from: " + args.model)
        chainer.serializers.load_npz(args.model, model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if args.gradient_clip > 0.:
        optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(args.gradient_clip))

    log(ID, "Loading training data")
    train = ModelDataset(dir=random_rollouts_dir, load_batch_size=args.load_batch_size, verbose=False)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=1, shuffle=False)

    updater = TBPTTUpdater(train_iter, optimizer, args.gpu, model.get_loss_func(), args.sequence_length)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output_dir)
    trainer.extend(extensions.snapshot(), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(10 if args.gpu >= 0 else 1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
    if not args.no_progress_bar:
        trainer.extend(extensions.ProgressBar(update_interval=10 if args.gpu >= 0 else 1))

    sample_size = 256
    rollout_z_t, rollout_z_t_plus_1, rollout_action, done = train[0]
    sample_z_t = rollout_z_t[0:sample_size]
    sample_z_t_plus_1 = rollout_z_t_plus_1[0:sample_size]
    sample_action = rollout_action[0:sample_size]
    img_t = vision.decode(sample_z_t).data
    img_t_plus_1 = vision.decode(sample_z_t_plus_1).data
    if args.predict_done:
        done = done.reshape(-1)
        img_t_plus_1[np.where(done[0:sample_size] >= 0.5), :, :, :] = 0 # Make done black
    save_images_collage(img_t, os.path.join(output_dir, 'train_t.png'))
    save_images_collage(img_t_plus_1, os.path.join(output_dir, 'train_t_plus_1.png'))
    image_sampler = ImageSampler(model.copy(), vision, args, output_dir, sample_z_t, sample_action)
    trainer.extend(image_sampler, trigger=(args.snapshot_interval, 'iteration'))

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
        image_sampler(trainer)

    log(ID, "Generating gif for a rollout generated in dream")
    if args.gpu >= 0:
        model.to_cpu()
    model.reset_state()
    # current_z_t = np.random.randn(64).astype(np.float32)  # Noise as starting frame
    rollout_z_t, rollout_z_t_plus_1, rollout_action, done = train[np.random.randint(len(train))]  # Pick a random real rollout
    current_z_t = rollout_z_t[0] # Starting frame from the real rollout
    current_z_t += np.random.normal(0, 0.5, current_z_t.shape).astype(np.float32)  # Add some noise to the real rollout starting frame
    all_z_t = [current_z_t]
    # current_action = np.asarray([0., 1.]).astype(np.float32)
    for i in range(rollout_z_t.shape[0]):
        # if i != 0 and i % 200 == 0: current_action = 1 - current_action  # Flip actions every 100 frames
        current_action = np.expand_dims(rollout_action[i], 0)  # follow actions performed in a real rollout
        output = model(current_z_t, current_action, temperature=args.sample_temperature)
        if args.predict_done:
            current_z_t, done = output
            done = done.data
            # print(i, current_action, done)
        else:
            current_z_t = output
        all_z_t.append(current_z_t.data)
        if args.predict_done and done[0] >= 0.5:
            break
    dream_rollout_imgs = vision.decode(np.asarray(all_z_t).astype(np.float32)).data
    dream_rollout_imgs = post_process_image_tensor(dream_rollout_imgs)
    imageio.mimsave(os.path.join(output_dir, 'dream_rollout.gif'), dream_rollout_imgs, fps=20)

    log(ID, "Done")


if __name__ == '__main__':
    main()