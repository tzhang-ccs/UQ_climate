import sys


labels = ['CNTL']
case_names = ['20231209.v3.LR.piControl-spinup.chrysalis']
case_prefix = '20231213.v3.LR.piControl-PPE.chrysalis'

for i in range(1,29):
    labels.append(f'M{i:03d}')
    case_names.append(f'{case_prefix}_M{i:03d}')

iice_base = 'https://portal.nersc.gov/project/m2136/bin/iice/iice.cgi?'
www_base = 'https://web.lcrc.anl.gov/public/e3sm/diagnostic_output/ac.wlin/E3SMv3_dev/20231213.v3.LR.piControl-PPE.chrysalis/'
url_base = f'e3sm_diags/atm_monthly_180x360_aave/model_vs_obs_0101-0110/viewer'

iice_link = iice_base
for i,l in enumerate(labels):
    idd = i+1
    iice_link = f'{iice_link}url{idd}={www_base}/{case_names[i]}/{url_base}&label{idd}={l}&'

iice_link = f'{iice_link}&category=&plot=&diff=0&dconfig=&'

print(iice_link)
