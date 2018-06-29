import argparse
import os
import time
from multiprocessing import cpu_count, Pool

import gzip
import gym
from scipy.misc import imresize
import numpy as np

from lib.utils import log, mkdir
from lib.constants import DOOM_GAMES
try:
    from lib.env_wrappers import ViZDoomWrapper
except Exception as e:
    None

ID = "random_rollouts"


def generate_action(low, high, prev_action, balance_no_actions=False, force_actions=False):
    if np.random.randint(10) % 10 and prev_action is not None:
        return (prev_action)

    action_len = len(low)
    action = [0 for i in range(action_len)]
    while True:
        for i in range(action_len):
            random = np.random.randint(low[i], high[i]+1)
            if random % action_len:
                action[i] = random
        # Because in many games all 0's or all 1's are the same action (no action), we want to limit
        # those to be the equal probability as all other combinations of actions. Maybe there's a
        # better way, or I'm not thinking of something:
        if balance_no_actions and (all(a == 0 for a in action) or all(a == 1 for a in action)) and np.random.randint(2)==0:
            action = [0 for i in range(action_len)]
            continue
        if force_actions and all(a == 0 for a in action):
            continue
        break

    return (np.array(action).astype(np.float32))


def worker(worker_arg_tuple):
    rollouts_per_core, args, output_dir = worker_arg_tuple

    np.random.seed()

    if args.game in DOOM_GAMES:
        env = ViZDoomWrapper(args.game)
    else:
        env = gym.make(args.game)

    for rollout_num in rollouts_per_core:
        t = 1

        actions_array = []
        frames_array = []
        rewards_array = []

        observation = env.reset()
        frames_array.append(imresize(observation, (args.frame_resize, args.frame_resize)))

        start_time = time.time()
        prev_action = None
        while True:
            # action = env.action_space.sample()
            action = generate_action(env.action_space.low,
                                     env.action_space.high,
                                     prev_action,
                                     balance_no_actions=True if args.game in DOOM_GAMES else False,
                                     force_actions=False if args.game in DOOM_GAMES else True)
            prev_action = action
            observation, reward, done, _ = env.step(action)
            actions_array.append(action)
            frames_array.append(imresize(observation, (args.frame_resize, args.frame_resize)))
            rewards_array.append(reward)

            if done:
                log(ID,
                    "\t> Rollout {}/{} finished after {} timesteps in {:.2f}s".format(rollout_num, args.num_rollouts, t,
                                                                                      (time.time() - start_time)))
                break
            t = t + 1

        actions_array = np.asarray(actions_array)
        frames_array = np.asarray(frames_array)
        rewards_array = np.asarray(rewards_array).astype(np.float32)

        rollout_dir = os.path.join(output_dir, str(rollout_num))
        mkdir(rollout_dir)

        # from lib.utils import post_process_image_tensor
        # import imageio
        # imageio.mimsave(os.path.join(output_dir, str(rollout_num), 'rollout.gif'), post_process_image_tensor(frames_array), fps=20)

        with gzip.GzipFile(os.path.join(rollout_dir, "frames.npy.gz"), "w") as file:
            np.save(file, frames_array)
        np.savez_compressed(os.path.join(rollout_dir, "misc.npz"),
                            action=actions_array,
                            reward=rewards_array)
        with open(os.path.join(rollout_dir, "count"), "w") as file:
            print("{}".format(frames_array.shape[0]), file=file)

    env.close()


def main():
    parser = argparse.ArgumentParser(description='World Models ' + ID)
    parser.add_argument('--data_dir', '-d', default="/data/wm", help='The base data/output directory')
    parser.add_argument('--game', default='CarRacing-v0',
                        help='Game to use')  # https://gym.openai.com/envs/CarRacing-v0/
    parser.add_argument('--experiment_name', default='experiment_1', help='To isolate its files from others')
    parser.add_argument('--num_rollouts', '-n', default=100, type=int, help='Number of rollouts to collect')
    parser.add_argument('--offset', '-o', default=0, type=int,
                        help='Offset rollout count, in case running on distributed cluster')
    parser.add_argument('--frame_resize', '-r', default=64, type=int, help='h x w resize of each observation frame')
    parser.add_argument('--cores', default=0, type=int, help='Number of CPU cores to use. 0=all cores')
    args = parser.parse_args()
    log(ID, "args =\n " + str(vars(args)).replace(",", ",\n "))

    output_dir = os.path.join(args.data_dir, args.game, args.experiment_name, ID)
    mkdir(output_dir)

    log(ID, "Starting")

    if args.cores == 0:
        cores = cpu_count()
    else:
        cores = args.cores
    start = 1 + args.offset
    end = args.num_rollouts + 1 + args.offset
    rollouts_per_core = np.array_split(range(start, end), cores)
    pool = Pool(cores)
    worker_arg_tuples = []
    for i in rollouts_per_core:
        if len(i) != 0:
            worker_arg_tuples.append((i, args, output_dir))
    pool.map(worker, worker_arg_tuples)
    pool.close()
    pool.join()

    log(ID, "Done")


if __name__ == '__main__':
    main()
