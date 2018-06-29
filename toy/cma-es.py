import os
import io
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from scipy.misc import imread

import numpy as np
import imageio

solution = [10, 10]


def schaffer(x, y):
    x -= solution[0]
    y -= solution[1]
    return 0.5 + ((np.sin((x ** 2) - (y ** 2)) ** 2) - 0.5) \
                 / \
                 ((1 + 0.001 * ((x ** 2) + (y ** 2))) ** 2)


def f(w):
    return -(schaffer(w[0], w[1]) - schaffer(solution[0], solution[1])) ** 2


def main():
    plt.switch_backend('agg')

    N = 2
    xmean = np.random.randn(N)
    sigma = 0.3
    stopeval = 1e3 * N ** 2
    stopfitness = 1e-10

    λ = 64  # 4+int(3*np.log(N))
    mu = λ // 4
    weights = np.log(mu + 1 / 2) - np.log(np.asarray(range(1, mu + 1))).astype(np.float32)
    weights = weights / np.sum(weights)
    mueff = (np.sum(weights) ** 2) / np.sum(weights ** 2)

    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, ((mueff - 1) / (N + 1)) ** 0.5 - 1) + cs

    pc = np.zeros(N).astype(np.float32)
    ps = np.zeros(N).astype(np.float32)
    B = np.eye(N, N).astype(np.float32)
    D = np.ones(N).astype(np.float32)

    C = B * np.diag(D ** 2) * B.T
    invsqrtC = B * np.diag(D ** -1) * B.T
    eigeneval = 0
    chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    counteval = 0
    generation = 0
    solution_found = False
    graphs = []
    while counteval < stopeval:
        arx = np.zeros((λ, N))
        arfitness = np.zeros(λ)
        for k in range(λ):
            arx[k] = xmean + sigma * B.dot(D * np.random.randn(N))
            arfitness[k] = f(arx[k])
            counteval += 1

        plt.ylim(-1, 20)
        plt.xlim(-1, 20)
        plt.plot(solution[0], solution[1], "b.")
        plt.plot(arx[:, 0], arx[:, 1], "r.")
        plt.plot(np.mean(arx[:, 0]), np.mean(arx[:, 1]), "g.")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.clf()
        buf.seek(0)
        img = imread(buf)
        buf.close()
        graphs.append(img)

        arindex = np.argsort(-arfitness)
        arfitness = arfitness[arindex]

        xold = xmean
        xmean = weights.dot(arx[arindex[0:mu]])

        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC.dot((xmean - xold) / sigma)
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / λ)) / chiN < 1.4 + 2 / (N + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
        artmp = (1 / sigma) * (arx[arindex[0:mu]] - xold)
        C = (1 - c1 - cmu) * C + c1 * (pc.dot(pc.T) + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp.T.dot(
            np.diag(weights)).dot(artmp)
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        if counteval - eigeneval > λ / (c1 + cmu) / N / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            D, B = np.linalg.eig(C)
            D = np.sqrt(D)
            invsqrtC = B.dot(np.diag(D ** -1).dot(B.T))

        generation += 1

        if arfitness[0] >= -stopfitness:
            solution_found = True
            break

    if solution_found:
        print("Solution found at generation #" + str(generation))
    else:
        print("Solution not found")

    if not os.path.exists("result"):
        os.makedirs("result")
    imageio.mimsave('result/cma-es.gif', graphs)


if __name__ == '__main__':
    main()
