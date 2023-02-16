import subprocess
import configargparse as argparse
import sys
from itertools import chain


parser = argparse.ArgumentParser()
parser.add_argument('--C_T0',type=float,default=0.5)
parser.add_argument('--C_s',type=float,default=0.5)
parser.add_argument('--C_l',type=float,default=1)
parser.add_argument('--rho',type=float,default=0.7)
parser.add_argument('--dgp',type=str,choices=['ar','arma','season_ar','season_arma','mix'],default='arma')
parser.add_argument('--dgp_p',type=int,default=1)
parser.add_argument('--dgp_r',type=int,default=4)
parser.add_argument('--dgp_s',type=int,default=1)
parser.add_argument('--task',type=str,choices=['rate','convergence'],required=True)
parser.add_argument('--N',type=int,default=10)
args = parser.parse_args()

rank = args.dgp_r + 2*args.dgp_s
if args.dgp == 'arma':
    if args.rho == 0.7 and args.dgp_r == 4 and args.N == 10:
        T_list = [370]*13
        T0_list = [28]*13
        base_s_list = [10]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-5,5)]))
    elif args.rho == 0.7 and args.dgp_r == 4 and args.N == 20:
        T_list = [1020,1460]*10
        T0_list = [47,57]*10
        base_s_list = [12,12]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-5,5)]))
    elif args.rho == 0.7 and args.dgp_r == 4 and args.N == 40:
        T_list = [1600,2100,3100,5600]*10
        T0_list = [60,68,83,112]*10
        base_s_list = [12,12,12,12]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-5,5)]))
    # elif args.rho == 0.5 and args.dgp_r == 4:
    #     T_list = [300,500,800,1700]
    #     T0_list = [25,33,42,61]
    #     s_list = [8,8,9,10]
    # elif args.rho == 0.3 and args.dgp_r == 4:
    #     T_list = [200,300,600,1000]
    #     T0_list = [21,25,36,47]
    #     s_list = [4,4,5,5]
    elif args.rho == 0.7 and args.dgp_r == 3:
        T_list = [500,800,1300,2800]*13
        T0_list = [33,42,54,79]*13
        base_s_list = [17,18,19,22]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-13,0)]))
    elif args.rho == 0.7 and args.dgp_r == 2:
        T_list = [300,500,800,1800]*13
        T0_list = [25,33,42,63]*13
        base_s_list = [15,17,18,20]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-13,0)]))
elif args.dgp == 'mix':
    if args.dgp_r == 4:
        T_list = [400,500,900,1900]*12
        T0_list = [30,33,45,65]*12
        base_s_list = [10,10,10,11]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-8,3)]))
    elif args.dgp_r == 3:
        T_list = [300,400,700,1400]*12
        T0_list = [25,30,39,56]*12
        base_s_list = [10,10,10,11]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-8,3)]))
    elif args.dgp_r == 2:
        T_list = [200,300,500,900]*12
        T0_list = [21,25,33,45]*12
        base_s_list = [10,10,10,10]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-8,3)]))

if args.task == 'rate':
    exp_name = f'{args.dgp}_choose_s_userate'


# for (p,r,dgp_s) in [(1,1,1),(0,1,1),(0,1,0),(0,0,1)]:
#     for T,T0,s in zip(T_list,T0_list,s_list):
#         for init in ['zero','true','spec']:
#             command = ['python3.9','run.py','--dgp_p',str(p),'--dgp_r',str(r),'--dgp_s',str(dgp_s),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--name',exp_name]
#             print(' '.join(command))
#             subprocess.run(command,stdout=None)

# rate
    for T,T0,s in zip(T_list,T0_list,s_list):
        # for init in ['GD']:
        #     command = ['python3.9','../train.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task',args.task]
        #     print(' '.join(command))
        #     subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        if float(sys.version[:3])<3.9:
            cmd = 'python'
        else:
            cmd = 'python3.9'
        for init in ['true']:
            command = [cmd,'../train.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task',args.task,'--stop_method','Adiff','--schedule','half','--lr','5e-4','--N',str(args.N)]
            print(' '.join(command))
            subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

# convergence
# elif args.task == 'convergence':
#     exp_name = 'convergence_test'
#     for T,T0,s,gamma in zip(T_list,T0_list,s_list,gamma_list):
#         for init in ['zero']:
#             command = ['python3.9','run.py','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--dgp_s',str(args.dgp_s),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--s_select','fnorm','--s_thres',str(gamma), '--visualize','--task',args.task]
#             print(' '.join(command))
#             subprocess.run(command,stdout=None)