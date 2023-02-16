import numpy as np
import tensorly as tl
from help_func import *
from DGP import *
from tensorOp import tensor_op


def ADMM(Y,X,A0,beta,lmbd,N,P,T,max_iter,true_A):
    """
    admm for low-rank matrices
    """
    C = np.zeros((N,N*P))

    # initial values
    A = A0
    W = np.copy(A)
    # est_err = np.linalg.norm(tensor_op.unfold(tensor_op.fold(A,(N,N,P),1) -true_A[:,:,:P],1),ord='fro')
    # Loss = np.sum(np.linalg.norm(Y - A @X,ord=2,axis=0)**2) / T + lmbd * np.linalg.norm(A,ord='nuc')
    # print("lmbd:",lmbd,"est err: ",est_err,"loss: ",Loss)


    for iter in range(max_iter):
        # exact update
        A = (np.linalg.inv(X @ X.T/(T-P) + beta*np.identity(N*P)) @ (Y/(T-P) @ X.T + beta*(W-C)).T).T
        u,s,v = np.linalg.svd(A+C,full_matrices=False)
        s = s - lmbd/(2*beta)
        s[s < 0] = 0
        W = u @ np.diag(s) @ v
        C = C + A - W

        # est_err = np.linalg.norm(tensor_op.unfold(tensor_op.fold(A,(N,N,P),1) -true_A[:,:,:P],1),ord='fro')
        # Loss = np.sum(np.linalg.norm(Y - A @X,ord=2,axis=0)**2) / T + lmbd * np.linalg.norm(A,ord='nuc')
        # print("lmbd:",lmbd,"est err: ",est_err,"loss: ",Loss)

        # stopping rule
        
    return A

def ADMM_mode2(Y,X,A0,beta,lmbd,N,P,T,max_iter,true_A):
    """
    admm for low-rank matrices
    """
    C = np.zeros((N,N*P))

    # initial values
    A = A0
    W = np.copy(A)

    for iter in range(max_iter):
        # exact update
        A = (np.linalg.inv(X @ X.T/(T-P) + beta*np.identity(N*P)) @ (Y/(T-P) @ X.T + beta*(W-C)).T).T
        u,s,v = np.linalg.svd(tensor_op.unfold(tensor_op.fold(A+C,(N,N,P),1),2),full_matrices=False)
        s = s - lmbd/(2*beta)
        s[s < 0] = 0
        W = u @ np.diag(s) @ v
        W = tensor_op.unfold(tensor_op.fold(W,(N,N,P),2),1).numpy()
        C = C + A - W

        # stopping rule
        
    return A

def ADMM_2(Y,X,A0,beta,lmbd,N,P,T,max_iter,true_A):
    """
    admm for low-rank tensors
    """
    C1 = np.zeros((N,N,P))
    C2 = np.zeros((N,N,P))

    # initial values
    A = tensor_op.fold(A0,(N,N,P),1).numpy()
    A_old = np.copy(A)
    W1 = np.copy(A)
    W2 = np.copy(A)
    stop_thres = 1e-5


    for iter in range(max_iter):
        # exact update
        A = (np.linalg.inv(X @ X.T/(T-P) + 2*beta*np.identity(N*P)) @ (Y @ X.T/(T-P) + beta*tensor_op.unfold(W1+W2-C1-C2,1).numpy()).T).T
        A = tensor_op.fold(A,(N,N,P),1).numpy()
        u,s,v = np.linalg.svd(tensor_op.unfold(A+C1,1),full_matrices=False)
        s = s - lmbd/(2*beta)
        s[s < 0] = 0
        W1 = u @ np.diag(s) @ v
        W1 = tensor_op.fold(W1,(N,N,P),1).numpy()
        C1 = C1 + A - W1
        u,s,v = np.linalg.svd(tensor_op.unfold(A+C2,2),full_matrices=False)
        s = s - lmbd/(2*beta)
        s[s < 0] = 0
        W2 = u @ np.diag(s) @ v
        W2 = tensor_op.fold(W2,(N,N,P),2).numpy()
        C2 = C2 + A - W2

        # stopping rule
        if max(np.linalg.norm(tensor_op.unfold(A-W1,1)),np.linalg.norm(tensor_op.unfold(A-W2,1)),np.linalg.norm(tensor_op.unfold(A-A_old,1))) < stop_thres*np.linalg.norm(A):
            return tensor_op.unfold(A,1).numpy()
        A_old = np.copy(A)

    return tensor_op.unfold(A,1).numpy()