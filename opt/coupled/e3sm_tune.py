import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import copy
import time
import os
import xgboost as xgb
import seaborn as sns
import xarray as xr
import sys
sys.path.append("/home/ac.tzhang/my_utils/src/")
from plot import contour
import cartopy.crs as ccrs
from global_land_mask import globe

from loguru import logger
logger.remove()
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
logger.add(sys.stdout, format=fmt)
log_path = f'tuning.log'
if os.path.exists(log_path):
    os.remove(log_path)
    ii = logger.add(log_path)

class diags():
    def __init__(self):
        self.varns = ['SWCF global ceres_ebaf_toa_v4.1','LWCF global ceres_ebaf_toa_v4.1','PRECT global GPCP_v2.3',
           'FLNS global ceres_ebaf_surface_v4.1','U-200mb global ERA5','PSL global ERA5','Z3-500mb global ERA5',
           'TREFHT land ERA5','T-200mb global ERA5','SST global HadISST_PI']
        self.shortname = ['SWCF','LWCF','PRECT','FLNS','U200','PSL','Z500','TREFHT','T200','SST']
        self.const = 'RESTOM global ceres_ebaf_toa_v4.1'
        self.const_shortname = 'Net TOA'
        self.all_varns = ['PRECT global GPCP_v2.3','FLUT global ceres_ebaf_toa_v4.1','FSNTOA global ceres_ebaf_toa_v4.1',
                          'SWCF global ceres_ebaf_toa_v4.1','LWCF global ceres_ebaf_toa_v4.1','NETCF global ceres_ebaf_toa_v4.1',
                          'FLNS global ceres_ebaf_surface_v4.1','FSNS global ceres_ebaf_surface_v4.1','LHFLX global ERA5',
                          'SHFLX global ERA5','PSL global ERA5','U-850mb global ERA5','U-200mb global ERA5',
                          'Z3-500mb global ERA5','T-850mb global ERA5','T-200mb global ERA5',
                          'TAUXY ocean ERA5','TREFHT land ERA5','SST global HadISST_PI']
        self.all_shortname = ['PRECT','FLUT','FSNTOA','SWCF','LWCF','NETCF','FLNS','FSNS','LHFLX','SHFLX',
                              'PSL','U850','U200','Z500','T850','T200','TAUXY','TREFHT','SST_PI']
        self.key_varns = ['SWCF global ceres_ebaf_toa_v4.1','LWCF global ceres_ebaf_toa_v4.1',
                          'PRECT global GPCP_v2.3','TREFHT land ERA5','PSL global ERA5','U-200mb global ERA5','U-850mb global ERA5',
                          'Z3-500mb global ERA5','SST global HadISST_PI']
        self.key_shortname = ['SW CRE','LW CRE','prec','tas land','SLP','u-200','u-850','Zg-500','SST']

        self.p_names = ['clubb_c1','clubb_gamma_coef','clubb_c_k10','zmconv_tau','zmconv_dmpdz', 'zmconv_micro_dcs',
            'nucleate_ice_subgrid','p3_nc_autocon_expon','p3_qc_accret_expon','zmconv_auto_fac',
            'zmconv_accr_fac','zmconv_ke','cldfrc_dp1','p3_embryonic_rain_size','effgw_oro']
        self.p_names = {'ocn':['config_redi_constant_kappa','config_gm_constant_kappa'],
                        'ice':['config_snow_thermal_conductivity','config_ice_ocean_drag_coefficient']}
        
        
        self.path_base = f"/home/ac.tzhang/fs0_large/E3SM_archive/"
        self.path_para = f'/lcrc/group/e3sm2/ac.wlin/E3SMv3/20231213.v3.LR.piControl-PPE.chrysalis/run/Allruns/'
        self.path_diag = f"/e3sm_diags/atm_monthly_180x360_aave/model_vs_obs_0101-0110/viewer/table-data/ANN_metrics_table.csv"
        self.path_e3sm = f'/lcrc/group/e3sm/ac.tzhang/E3SMv3/'
        self.path_tune = f'/home/ac.tzhang/E3SM/UQ_climate/opt/coupled'
        self.path_www  = f'/home/ac.tzhang/www/e3sm_diags/'
        self.path_archive =  f'/home/ac.tzhang/e3sm2_large/tune_20231222/'
        self.archive_id = 3

        self.get_cntl_score()
    
    def get_cntl_score(self):
        cntl_name = f'20231209.v3.LR.piControl-spinup.chrysalis'
        case_path = '/home/ac.wlin/www/E3SMv3_dev/20231213.v3.LR.piControl-PPE.chrysalis/'
        data_diags = pd.read_csv(f'{case_path}/{cntl_name}/{self.path_diag}').set_index('Variables')
        data_diags = data_diags.loc[self.key_varns]
        self.rmse_cntl = data_diags['RMSE']

    def get_score(self,case_path,case_name):
        data_diags = pd.read_csv(f'{case_path}/{case_name}/{self.path_diag}').set_index('Variables')
        const = data_diags['Test_mean'].loc[self.const]
        data_diags = data_diags.loc[self.key_varns]
        score = data_diags['RMSE']/self.rmse_cntl
        score.columns = self.key_shortname
        score['metrics'] = np.sum(score.values) + const
        return score

    def get_scoreset(self, inc_tuning=True):
        self.scoreset = pd.DataFrame()
        case_prefix = '20231213.v3.LR.piControl-PPE.chrysalis'
        case_path = '/home/ac.wlin/www/E3SMv3_dev/20231213.v3.LR.piControl-PPE.chrysalis/'
        for i in range(1,29):
            cn = f'{case_prefix}_M{i:03d}'
            tmp = self.get_score(case_path,cn)
            self.scoreset[f'M{i:03d}'] = tmp

        if inc_tuning:
            case_name = '20231213.v3.LR.piControl-PPE.tune.chrysalis'
            for i in range(1,self.archive_id):
                case_path = f'{self.path_archive}/T{i:03d}/'
                tmp = self.get_score(case_path,case_name)
                self.scoreset[f'T{i:03d}'] = tmp
        
        self.scoreset = self.scoreset.T
        self.scoreset.columns = self.key_shortname + ['metrics']
        return self.scoreset

    def get_parameters(self,inc_tuning=True):
        case_prefix='20231213.v3.LR.piControl-PPE.chrysalis'
        
        paraset = pd.DataFrame()
        for i in range(1,29):
            ocn_path = f'{self.path_para}/{case_prefix}_M{i:03d}/mpaso_in'
            ice_path = f'{self.path_para}/{case_prefix}_M{i:03d}/mpassi_in'

            tmp = {}
            for c in self.p_names:
                for pn in self.p_names[c]:
                    if c == 'ocn':
                        var_str = os.popen(f'grep -w {pn} {ocn_path}').read().split('=')[1]
                    if c == 'ice':
                        var_str = os.popen(f'grep -w {pn} {ice_path}').read().split('=')[1]
                    var = float(var_str)
                    tmp[pn] = var

            tmp = pd.DataFrame(tmp,index=[f'M{i:03d}'])
            paraset = pd.concat([paraset,tmp])

        if inc_tuning:
            for i in range(1,self.archive_id):
                ocn_path = f'{self.path_archive}/T{i:03d}/mpaso_in'
                ice_path = f'{self.path_archive}/T{i:03d}//mpassi_in'
    
                tmp = {}
                for c in self.p_names:
                    for pn in self.p_names[c]:
                        if c == 'ocn':
                            var_str = os.popen(f'grep -w {pn} {ocn_path}').read().split('=')[1]
                        if c == 'ice':
                            var_str = os.popen(f'grep -w {pn} {ice_path}').read().split('=')[1]
                        var = float(var_str)
                        tmp[pn] = var
    
                tmp = pd.DataFrame(tmp,index=[f'T{i:03d}'])
                paraset = pd.concat([paraset,tmp])

        return paraset

    def run_model(self, para):
        case_prefix='20231213.v3.LR.piControl-PPE.tune.chrysalis'
        mesg = ''
        
        para = para[0,:]

        os.chdir(f'{self.path_e3sm}/{case_prefix}/case_scripts/')
        #ocn
        paras = {}
        for i,n in enumerate(self.p_names['ocn']):
            paras[n] = para[i]
            mesg = f'{mesg} {n}={para[i]:.5e}'

        for key in paras:
            replace_str = "sed -i '/\<"+key+"\>/c\ "+key+"="+str(paras[key])+"' user_nl_mpaso"
            os.system(replace_str)
            
        #ice
        paras = {}
        for i,n in enumerate(self.p_names['ice']):
            paras[n] = para[i+len(self.p_names['ocn'])]
            mesg = f'{mesg} {n}={paras[n]:.5e}'

        for key in paras:
            replace_str = "sed -i '/\<"+key+"\>/c\ "+key+"="+str(paras[key])+"' user_nl_mpassi"
            os.system(replace_str)


        # run case
        os.system("pwd")
        os.system("./case.submit >& case_id")
        jid = os.popen("tail -n 1 case_id |awk '{print $6}'").read().strip()
        logger.debug(f'CaseID={self.archive_id},Submit E3SM with job id {jid}')

        while os.popen("squeue -u ac.tzhang").read().find(jid) != -1:
            time.sleep(60)

        logger.debug(f'Finish E3SM with job id {jid}')
        os.system('./case.st_archive > /dev/null')
        
        # get metrics score
        os.chdir(self.path_tune)
        os.system('zppy -c post.20231209.v3.LR.piControl-spinup.chrysalis.cfg > /dev/null')
        score = self.get_score(self.path_www,case_prefix)['metrics']

        mesg = f'{mesg} : score={score:.3f}'
        logger.info(mesg)

        #backup

        os.system(f'mkdir -p {self.path_archive}/T{self.archive_id:03d}')
        os.system(f'mv {self.path_e3sm}/{case_prefix}/archive/rest {self.path_archive}/T{self.archive_id:03d}')
        os.system(f'mv {self.path_e3sm}/{case_prefix}/diag/post/ {self.path_archive}/T{self.archive_id:03d}')
        os.system(f'cp {self.path_e3sm}/{case_prefix}/case_scripts/user_nl_* {self.path_archive}/T{self.archive_id:03d}')
        os.system(f'cp {self.path_e3sm}/{case_prefix}/run/*in {self.path_archive}/T{self.archive_id:03d}')
        os.system(f'mv {self.path_www}/{case_prefix}/ {self.path_archive}/T{self.archive_id:03d}')
        
        self.archive_id = self.archive_id + 1
      
