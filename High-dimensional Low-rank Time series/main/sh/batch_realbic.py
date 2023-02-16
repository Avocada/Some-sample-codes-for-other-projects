import subprocess
import configargparse as argparse


parser = argparse.ArgumentParser()
parser.add_argument('--rho',type=float,default=0.7)
parser.add_argument('--dgp',type=str,choices=['ar','arma','season_ar','season_arma','mix'],default='arma')
parser.add_argument('--dgp_p',type=int,default=1)
parser.add_argument('--dgp_r',type=int,default=4)
parser.add_argument('--dgp_s',type=int,default=1)
args = parser.parse_args()

for T0 in [8,9,10]:
    command = ['python3.9','../real_bic.py','--data','RV','--candidate_s','1','2','3','--candidate_r1','1','2','3','--candidate_r2','1','2','3','--T0',str(T0),'--T0_type','fix','--bic_c','2e-2']
    print(' '.join(command))
    subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)