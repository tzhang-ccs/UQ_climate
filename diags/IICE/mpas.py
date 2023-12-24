labels = ['CNTL',
          'ocn.0',
          'ocn.1',
          'ocn.2',
          'ocn.3']

case_names = ['cntl',
              'workdir.ocn.0',
              'workdir.ocn.1',
              'workdir.ocn.2',
              'workdir.ocn.3']

iice_base = 'https://portal.nersc.gov/project/m2136/bin/iice/mpas-a.cgi?'
www_base = 'https://web.lcrc.anl.gov/public/e3sm/diagnostic_output/ac.tzhang'
url_base = f'mpas_analysis/ts_0001-0010_climo_0001-0010'

iice_link = iice_base
for i,l in enumerate(labels):
    idd = i+1
    if l == 'CNTL':
        e3sm_name = '20230223.NGD_v3atm.piControl.tune'
    else:
        e3sm_name = '20230223.NGD_v3atm.piControl.ocn.tune'
    iice_link = f'{iice_link}url{idd}={www_base}/{case_names[i]}/{e3sm_name}/{url_base}&label{idd}={l}&'

iice_link = f'{iice_link}&category=ocean&component=ocean&plot=&diff=0&dconfig=wcycl&'

print(iice_link)
