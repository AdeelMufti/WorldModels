import matplotlib.pyplot as plt

plt.switch_backend('agg')

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, report
from chainer.training import extensions

import numpy as np


class MDN(chainer.Chain):
    def __init__(self, hidden_dim, output_dim, k):
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.k = k
        super(MDN, self).__init__(
            input_layer=L.Linear(None, hidden_dim),
            coef_layer=L.Linear(hidden_dim, k * output_dim),
            mu_layer=L.Linear(hidden_dim, k * output_dim),
            ln_var_layer=L.Linear(hidden_dim, k * output_dim),
        )

    def __call__(self, input):
        coef, mu, ln_var = self.fprop(input)

        def sample(row_num):
            cum_prod = 0
            r = np.random.uniform()
            index = None
            for i, probability in enumerate(coef[row_num]):
                cum_prod += sum(probability)
                if r <= cum_prod.data:
                    index = i
                    break

            return F.gaussian(mu[row_num][index], ln_var[row_num][index])

        output = F.expand_dims(sample(0), 0)
        for row_num in range(1, input.shape[0]):
            this_output = F.expand_dims(sample(row_num), 0)
            output = F.concat((output, this_output), axis=0)

        return output

    def fprop(self, input):
        k = self.k
        output_dim = self.output_dim

        h = self.input_layer(input)

        coef = F.softmax(self.coef_layer(h))
        mu = self.mu_layer(h)
        ln_var = self.ln_var_layer(h)

        mu = F.reshape(mu, (-1, k, output_dim))
        coef = F.reshape(coef, (-1, k, output_dim))
        ln_var = F.reshape(ln_var, (-1, k, output_dim))

        return coef, mu, ln_var

    def get_loss_func(self):
        def lf(input, output, epsilon=1e-8):
            output_dim = self.output_dim

            coef, mu, ln_var = self.fprop(input)

            output = F.reshape(output, (-1, 1, output_dim))
            mu, output = F.broadcast(mu, output)

            var = F.exp(ln_var)

            density = F.sum(
                coef *
                (1 / (np.sqrt(2 * np.pi) * F.sqrt(var))) *
                F.exp(-0.5 * F.square(output - mu) / var)
                , axis=1)

            nll = -F.sum(F.log(density))
            report({'loss': nll}, self)
            return nll

        return lf


class Linear(chainer.Chain):
    def __init__(self, hidden_dim, output_dim):
        self.output_dim = output_dim
        super(Linear, self).__init__(
            input_layer=L.Linear(None, hidden_dim),
            output_layer=L.Linear(hidden_dim, output_dim),
        )

    def __call__(self, input):
        return self.fprop(input)

    def fprop(self, input):
        h = self.input_layer(input)
        return self.output_layer(h)

    def get_loss_func(self):
        def lf(input, output):
            pred = self.fprop(input)
            loss = F.mean_squared_error(output.reshape(-1, 1), pred)
            report({'loss': loss}, self)
            return loss

        return lf


def main():
    model = MDN(256, 1, 5)
    # model = Linear(256, 1)

    points = 500

    y = np.random.rand(points).astype(np.float32)
    x = np.sin(2 * np.pi * y) + 0.2 * np.random.rand(points) * (np.cos(2 * np.pi * y) + 2)
    x = x.astype(np.float32)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    dataset = datasets.tuple_dataset.TupleDataset(x.reshape(-1, 1), y)
    train_iter = iterators.SerialIterator(dataset, batch_size=100)
    updater = training.StandardUpdater(train_iter, optimizer, loss_func=model.get_loss_func())
    trainer = training.Trainer(updater, (2000, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
    trainer.run()

    plt.ylim(-0.1, 1.1)
    plt.plot(x, y, "b.")
    plt.savefig("result/mdn-data_only.png")
    plt.clf()

    x_test = np.linspace(min(x), max(x), points).astype(np.float32)
    y_pred = model(x_test.reshape(-1, 1)).data

    plt.ylim(-0.1, 1.1)
    plt.plot(x, y, "b.")
    plt.plot(x_test, y_pred, "r.")
    plt.savefig("result/mdn-with_preds.png")


if __name__ == '__main__':
    main()
