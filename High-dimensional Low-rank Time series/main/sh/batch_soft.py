import subprocess
import configargparse as argparse


parser = argparse.ArgumentParser()
parser.add_argument('--rho',type=float,default=0.7)
parser.add_argument('--dgp',type=str,choices=['ar','arma','season_ar','season_arma','mix'],default='arma')
parser.add_argument('--dgp_p',type=int,default=1)
parser.add_argument('--dgp_r',type=int,default=4)
parser.add_argument('--dgp_s',type=int,default=1)
args = parser.parse_args()

rank = args.dgp_r + 2*args.dgp_s
if args.dgp == 'arma':
    T = 800
    T0_list = [25,50,100,200]
    if args.rho == 0.7 and args.dgp_r == 4:
        s_list = [10,10,10,10]
elif args.dgp == 'season_ar':
    args.dgp_p = 4
    T = 600
    T0_list = [9,36,63,99]
    s_list = [5,5,5,5]
elif args.dgp == 'mix':
    T = 1000
    T0_list = [20,40,60,80]
    if args.dgp_r == 4:
        s_list = [8,8,8,8]
    elif args.dgp_r == 3:
        s_list = [9,10,10,11]
    elif args.dgp_r == 5:
        s_list = [7,9,9,10]


exp_name = f'soft_{args.dgp}_1219'

for T0,s in zip(T0_list,s_list):
    for init in ['noisetrue']:
        command = ['python3.9','../aic.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--true_s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--stop_method','loss','--schedule','none','--lr','1e-3','--aic_c','0.07','--N','20']
        print(' '.join(command))
        subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    # for init in ['noisetrue']:
    #     command = ['python3.9','../cv.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--stop_method','loss','--schedule','none','--lr','5e-4','--thresholding_interval','1']
    #     print(' '.join(command))
    #     subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)