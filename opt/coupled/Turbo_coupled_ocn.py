import numpy as np
import xarray as xr
import scipy.stats
import os
import time
import pandas as pd
import xarray as xr
import warnings
from e3sm_tune import diags
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/ac.tzhang/E3SM/UQ_climate/opt/coupled/TuRBO/')
from turbo import Turbo1

from loguru import logger
logger.remove()
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
logger.add(sys.stdout, format=fmt)
log_path = f'tuning.log'
if os.path.exists(log_path):
    os.remove(log_path)
ii = logger.add(log_path)

dd = diags()
scoreset = dd.get_scoreset()
paraset = dd.get_parameters()

class E3SM:
    def __init__(self, dim=4):
        self.dim = dim
        self.lb = np.array([350,1.0,0.1,1800,-2.0e-3])
        self.lb = np.array([1.0, 0.1, 0.3, 1800, -1e-3,  100e-6, 1,  -1.8, 1.1, 5.0, 1.5, 0.5e-6, 0.01, 15e-6, 0.3])
        self.lb = np.array([400,400,0.03,0.2e-3])
        self.ub = np.array([1800,1800,0.65,160e-3])
        #self.ub = np.array([1400,5.0,0.5,14400,-0.1e-3])
        #self.ub = np.array([1400,5.0,0.5,5000,-0.1e-3])
        #self.ub = np.array([3.0, 0.3, 0.75,8100, -0.1e-3,250e-6, 1.4,-1.2, 1.3, 7.5, 2.0, 5.0e-6, 0.05, 30e-6, 0.4])
        
    def __call__(self, x):
        x = x.reshape(1,-1)
        val = dd.run_model(x)
        return val

f = E3SM()

turbo1 = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Latin hypercube design
    max_evals = 500,  # Maximum number of evaluations
    batch_size=8,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)


turbo1.optimize(scoreset,paraset, sampling_flag=False)
sys.exit()

X = turbo1.X  # Evaluated points
fX = turbo1.fX  # Observed values

np.save('paras',X)
np.save('score',fX)
