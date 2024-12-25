import numpy as np
import pandas as pd
import xarray as xr
import warnings
import json
import os
import sys
from scipy.stats import qmc
import multiprocessing as mp
warnings.filterwarnings("ignore")
import time

from loguru import logger
logger.remove()
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
logger.add(sys.stdout, format=fmt)
log_path = f'tuning.log'
if os.path.exists(log_path):
    os.remove(log_path)
ii = logger.add(log_path)

class uq:
    def __init__(self,name):
        self.config_paras(name)

    def config_paras(self, name):
        with open(name, 'r') as file:
            data = json.load(file)

        self.para_names = list(data.keys())
        self.lbs = [data[k][0] for k in data]
        self.ubs = [data[k][1] for k in data]
        self.dim = len(self.para_names)

        print(self.para_names)
        print(self.lbs)
        print(self.ubs)

    def lhs_sample(self,num, continue_run=False,continue_id=0):
        if continue_run:
            tmp = np.load('lhs_sample.npy')
            self.sample_scaled = tmp[continue_id:,:]
        else:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=num)
            self.sample_scaled = qmc.scale(sample, self.lbs, self.ubs)
            np.save('lhs_sample',self.sample_scaled)

    def run_case(self,id,data):
        print(f'{id=}, {data=}')
      
        if id % 2 == 0:
            model_path = '/lcrc/group/e3sm/ac.tzhang/E3SMv3/20241006.v3.LR.F2010_tuning/case_scripts'
        else:
            model_path = '/lcrc/group/e3sm/ac.tzhang/E3SMv3/20241006_tuning_extra/20241006.v3.LR.F2010_tuning/case_scripts'
        os.chdir(model_path)

        #config parameter set to replace the values
        para_set = {}
        mesg = ''
        for i,n in enumerate(self.para_names):
            para_set[n] = data[i]
            mesg = f'{mesg} {n}={data[i]:.5e}'

        print(para_set)

        # replace the values
        for k in para_set:
            rep_command = "sed -i '/\<"+k+"\>/c\ "+k+"="+str(para_set[k])+"' user_nl_eam"
            os.system(rep_command)

        # run the model
        os.system("./case.submit >& case_id")
        jid = os.popen("tail -n 1 case_id |awk '{print $6}'").read().strip()

        logger.debug(f'CaseID={id},Submit E3SM with job id {jid}')

        while os.popen("squeue -u ac.tzhang").read().find(jid) != -1:
            time.sleep(60)

        logger.debug(f'Finish E3SM with job id {jid}')

        self.archive_model(id)

    def archive_model(self,id):
        os.system('./case.st_archive >& /dev/null')
        os.system('zppy -c post.v3.LR.F2010.cfg >& /dev/null')

        base_path = '/lcrc/group/e3sm/ac.tzhang/tune_atm_20241008/'
        target_path = f'{base_path}/S{id:03d}'

        os.system(f'mkdir -p {target_path}')
        os.system(f'mv ../diag/post {target_path} ')
        os.system(f'cp user_nl_eam ../run/atm_in {target_path}')
      
    def analyse(self, method):
        if method == 'sample':
            pool = mp.Pool(2)
            pool.starmap(self.run_case, [(i+1,d) for i,d in enumerate(self.sample_scaled)])
        
        elif method == 'tune':
            self.run_case(0,self.sample_scaled[0,:])


