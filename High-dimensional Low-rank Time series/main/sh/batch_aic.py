import subprocess
import configargparse as argparse


parser = argparse.ArgumentParser()
parser.add_argument('--rho',type=float,default=0.7)
parser.add_argument('--dgp',type=str,choices=['ar','arma','season_ar','season_arma','mix'],default='arma')
parser.add_argument('--dgp_p',type=int,default=1)
parser.add_argument('--dgp_r',type=int,default=4)
parser.add_argument('--dgp_s',type=int,default=1)
parser.add_argument('--N',type=int,default=10)
args = parser.parse_args()

rank = args.dgp_r + 2*args.dgp_s
if args.dgp == 'arma':
    T_list = [500,1000,1500,2000]
    T0_list = [33,47,58,67]
    if args.rho == 0.7 and args.dgp_r == 4:
        s_list = [10,11,12,12]
    elif args.rho == 0.7 and args.dgp_r == 3:
        s_list = [10,11,12,12]
    elif args.rho == 0.7 and args.dgp_r == 5:
        s_list = [9,10,11,12]
elif args.dgp == 'season_ar':
    args.dgp_p = 4
    T_list = [200,400,600]
    T0_list = [9,9,9]
    s_list = [5,5,5]


exp_name = f'{args.dgp}_1214'

for T,T0,s in zip(T_list,T0_list,s_list):
    for init in ['noisetrue']:
        if T == 1500:
            command = ['python3.9','../aic.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--true_s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--stop_method','loss','--schedule','none','--lr','5e-4','--aic_c','0.03','--N',str(args.N)]
            print(' '.join(command))
            subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)