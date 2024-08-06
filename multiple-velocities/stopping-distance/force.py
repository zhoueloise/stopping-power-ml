#!/usr/bin/env python
# coding=utf-8
import logging
from tqdm import tqdm
from glob import glob
import numpy as np
import argparse
import os
import sys
sys.path.append(f"/scratch/users/elzhou2/YF_git_clone")
import pickle as pkl
import pandas as pd

from stopping_power_ml.stop_distance import StoppingDistanceComputer
from stopping_power_ml.integrator import TrajectoryIntegrator
import keras
import h5py 
import os
import time
import functools
from datetime import datetime
print = functools.partial(print, flush=True)

# Set up the root logger to INFO to suppress DEBUG messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Make an argument parser
parser = argparse.ArgumentParser(description="Run a stopping distance simulation in a single direction")
parser.add_argument('--direction', nargs = 3, type = int, default=[1, 0, 0], help='Direction vector')
parser.add_argument('--random-dir', action = 'store_true', help = 'Projectiles move in a random direction')
parser.add_argument('--random-seed', default = 1, type = int)
parser.add_argument('--model', default = "../model-random-and-channel.h5", type = str, help = "machine learned model")
#parser.add_argument('velocity', help='Starting velocity magnitude', type=float)

# Parse the arguments
args = parser.parse_args()
logging.info("Arguments received:")
for arg in vars(args):
    logging.info(f"{arg}: {getattr(args, arg)}")

mpath = args.model
model = keras.models.load_model(mpath)

with open(os.path.join('..', 'featurizer.pkl'), 'rb') as fp:
    featurizers = pkl.load(fp)

start_frame = pkl.load(open(os.path.join('..', '..', 'al_starting_frame.pkl'), 'rb'))
traj_int = TrajectoryIntegrator(start_frame, model, featurizers)

computer = StoppingDistanceComputer(traj_int)


# Generate random starting points and directions on the unit sphere
# rng = np.random.RandomState(args.random_seed)
seed = args.random_seed
rng = np.random.RandomState(seed)
vmag = 4
if not args.random_dir:
    velocity = np.array(args.direction, dtype=float)
    velocity *= vmag / np.linalg.norm(velocity)
else:
    u = rng.uniform(-1, 1)
    v = rng.uniform(0, 2 * np.pi)
    velocities = np.hstack((
        np.sqrt(1 - u ** 2) * np.cos(v),
        np.sqrt(1 - u ** 2) * np.sin(v),
        u
    )) * args.velocity

#position = rng.uniform(size = 3)
position = np.array([0.0, 0.75, 0.75])

force_calc = traj_int.create_force_calculator_given_displacement(position, velocity)

x = np.linspace(0, 200, 2000)
v = np.linspace(0.5, vmag, 40, endpoint = True)

X, V = np.meshgrid(x, v, indexing='ij')

X, V = X.flatten(), V.flatten()

# sample a gaussian distribution in the y z plane, and put that in a 3D array, called variance 
mean = [0,0]
cov = [[0.24, 0],[0, 0.24]]
size = 80000
ycoord, zcoord = rng.multivariate_normal(mean, cov, size).T
xcoord = [0] * size
variance = np.column_stack((xcoord,ycoord,zcoord))

print(X.shape)
print(V.shape)
print(variance.shape)

states = []
for i in tqdm(range(X.shape[0])):
    f = force_calc(X[i], V[i], variance[i])
    states.append([X[i], variance[i].item(0), variance[i].item(1), variance[i].item(2), V[i], f])

print(states[0])

states = np.vstack(states).reshape([x.shape[0], -1, 6])
print(states[0], states[1])

jobid = os.getenv('SLURM_JOBID', '')

output_dir = f'all_force/{jobid}'

os.makedirs(output_dir, exist_ok=True)

with h5py.File(f'{output_dir}/force.h5', 'w') as h5file:
    # Create a dataset with compression
    dataset = h5file.create_dataset('states', data = states, compression='gzip', compression_opts=4)

    # Add attributes
    info = {'start_pos': position, 
            'start_vel': velocity, 
            "model": mpath, 
            "shape": "n_disp, n_vmag, 6",
            "distribution": "multivariate normal",
            "col": "disp, vmag, force",
            "random_seed": seed,
            "mean": mean,
            "covariance": cov,
            "timestamp": datetime.now().isoformat()}
    for i in info.keys():
        dataset.attrs[i] = info[i]
