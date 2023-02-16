"""
Alternating least squares for omega and S, U1, U2
"""

import numpy as np
from numpy.random.mtrand import gamma
import scipy
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker as kron
from tensorly.tenalg import mode_dot
from help_func import *

def ALS(y,p,r,s,r1,r2,N,T,P,n_iter,lr_omega,lmbd_true,gamma_true,theta_true,true_A,true_G,stop_thres,flag_true_G,stop_method):
    # Initialization
    d = p+r+2*s
    Loss = np.inf
    S,U1,U2 = spectral_init_A_exp(y,N,T,P,r1,r2)
    # A = init_A(y,N,T,P,r1,r2)
    lmbd = np.copy(lmbd_true[:])
    gamma = np.copy(gamma_true[:])
    theta = np.copy(theta_true[:])
    # truncated L
    L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    S = get_G(S,L)
    G = mode_dot(mode_dot(S,U1,0),U2,1)
    # full L
    L = get_L(lmbd,gamma,theta,N,r,s,T,p)
    # use full L and G to restore a full A
    A = mode_dot(G,L,2)
    if flag_true_G:
        S,U1,U2 = get_S_and_U(true_G[:,:,:d],r1,r2,d)
        G = mode_dot(S,np.array(U1),0)
        G = mode_dot(G,np.array(U2),1)

    # useful for ls
    Y = y[:,1:]
    # Y_col = np.reshape(np.flip(Y,axis=1),(-1,1),order='F') # vectorized Y
    Y_col = np.reshape(Y,(-1,1),order='F') # y2 to yt
    n = T-1
    x = np.reshape(np.flip(y,axis=1),(-1,1),order='F') # vectorized y, yt to y1
    X1 = np.zeros((N*(T-1),T-1))
    for i in range(T-1):
        X1[:(i+1)*N,i:i+1] = x[(T-i-1)*N:]
    X2 = np.zeros((T-1,T-1,N))
    if p == 2:
        for i in range(1,T-1):
            X2[i,:(i+1-p),:] = np.flip(y[:,:(i+1-p)],axis=1).T
    else:
        for i in range(T-1):
            X2[i,:(i+1-p),:] = np.flip(y[:,:(i+1-p)],axis=1).T

    # loss(y,G,L,T)
    # loss_vec(Y,X1,G,L,T)
    # ALS steps
    flag_maxiter = 0
    for iter_no in range(n_iter):
        pre_lmbd = np.copy(lmbd)
        for k in range(r):
            # update lmbd
            grad,hess =vec_jac_hess_lmbd(lmbd[k],k,G,L,Y,X1,X2,p,T)
            # grad1,hess1 = jac_hess_lmbd(lmbd[k],k,G,L,y,p,T)
            # lmbd[k] = lmbd[k] - lr_omega * jac_lmbd(lmbd[k],k,G,L,y,p,T) / hess_lmbd(lmbd[k],k,G,L,y,p,T)
            temp = lmbd[k] - lr_omega * grad / hess
            if temp > 0.9 or temp < -0.9:
                temp = lmbd[k] - 0.1*lr_omega * grad
            lmbd[k] = max(min(0.9,temp),-0.9)
            power_series = np.arange(1,T-p+1)
            L[p:,p+k] = np.power(lmbd[k],power_series)
            # print("grad: ",grad)
            # print('hess: ',hess)
            # Loss = loss(y,G,L,T)
            # print('lmbd: ',Loss)
        pre_gamma = np.copy(gamma)
        pre_theta = np.copy(theta)
        for k in range(s):
            # update gamma and theta
            grad_gamma,grad_theta,hess = vec_jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,Y,X1,X2,p,r,T)
            # grad_gamma,grad_theta,hess = jac_hess_gamma_theta([gamma[k],theta[k]],k,G,L,y,p,r,T)
            grad = np.array([grad_gamma,grad_theta])
            hess_inv = np.linalg.inv(hess)
            temp = gamma[k] - lr_omega * (hess_inv @ grad)[0]
            if temp > 0.9 or temp < 0.1:
                temp = gamma[k] - lr_omega * grad_gamma
                theta[k]= theta[k] - lr_omega * grad_theta
            else:
                theta[k]= theta[k] - lr_omega * (hess_inv @ grad)[1]
            gamma[k] = max(min(0.9,temp),0.1)
            theta[k] = max(min(np.pi/2,theta[k]),-np.pi/2)
            power_series = np.arange(1,T-p+1)
            L[p:,p+r+2*k] = np.einsum('i,i->i',np.power(gamma[k],power_series),np.cos(power_series*theta[k]))
            L[p:,p+r+2*k+1] = np.einsum('i,i->i',np.power(gamma[k],power_series),np.sin(power_series*theta[k]))
        
        # print(lmbd)
        # print(gamma,theta)
        # prepare L, z, Z
        # note that there are in total T-1 z_t
        # L = get_L(lmbd,gamma,theta,N,r,s,T,p)
        # z = np.zeros((N*d,T-1))
        # for t in range(1,T):
        #     x_t = x[(T-t)*N:] 
        #     z[:,t-1:t] = get_z(x_t,N,L[:t,:])

        z = kron([L[:T-1,:].T,np.identity(N)]) @ X1

        S1 = tensor_op.unfold(S,1)

        # update U1
        X_1 = S1 @ kron([np.identity(d),U2.T]) @ z
        U1 = (X_1 @ Y.T).T @ np.linalg.pinv(X_1 @ X_1.T)
        # print("U1's norm: ",np.linalg.norm(U1,ord='fro'))

        # update U2
        X_2 = np.zeros((n*N,N*r2))
        # Z = get_Z(z,N)
        for i in range(n):
            X_2[i*N:(i+1)*N,:] = U1 @ S1 @ kron([np.reshape(z[:,i],(N,d),order='F').T ,np.identity(r2)])
        U2 = np.reshape((Y_col.T @ X_2) @ np.linalg.pinv(X_2.T @ X_2), (r2,N),order='F').T
        # print("U2's norm: ",np.linalg.norm(U2,ord='fro'))

        # update S
        X_s = np.zeros((n*N,r1*r2*d))
        for i in range(n):
            X_s[i*N:(i+1)*N,:] = kron([z[:,i].T @ kron([np.identity(d), U2]), U1])
        S = np.reshape(np.linalg.pinv(X_s.T @ X_s) @ (X_s.T @ Y_col), (r1,r2,d),order='F')
        # print("S's norm: ",np.linalg.norm(tensor_op.unfold(S,1),ord='fro'))

        # restore G
        pre_G = G
        G = mode_dot(S,np.array(U1),0)
        G = tl.tenalg.mode_dot(G,np.array(U2),1)
        # print("G's norm: ",np.linalg.norm(tensor_op.unfold(G,1),ord='fro'))

        # HOSVD to make U1 U2 orthonormal
        S,U1,U2 = get_S_and_U(G,r1,r2,d)

        # early stop
        pre_A = A
        A = tl.tenalg.mode_dot(G,L,2)
        # pre_loss = Loss
        # Loss = loss(y,G,L,T)
        # if pre_loss-Loss < 0:
        #     Loss = loss(y,G,L,T)
        #     print(p,r,s," Loss increase, No. of iter: ", iter_no)
        #     # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
        #     # print("Final loss: ", Loss)
        #     return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro') < stop_thres * np.linalg.norm(tensor_op.unfold(G,1), ord='fro')) and (np.linalg.norm(np.concatenate([lmbd-pre_lmbd,gamma-pre_gamma,theta-pre_theta]), ord=2) < stop_thres * np.linalg.norm(np.concatenate([lmbd,gamma,theta]), ord=2)):
        if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro') < stop_thres) and (np.linalg.norm(np.concatenate([lmbd-pre_lmbd,gamma-pre_gamma,theta-pre_theta]), ord=2) < 0.1*stop_thres): # original simulation result
        # if (stop_method == 'SepEst') and (np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro') < stop_thres * np.linalg.norm(tensor_op.unfold(G,1), ord='fro')) and (np.linalg.norm(np.concatenate([lmbd-pre_lmbd,gamma-pre_gamma,theta-pre_theta]), ord=2) < stop_thres * np.linalg.norm(np.concatenate([lmbd,gamma,theta]), ord=2)): # 11。25: for 001 simulation
            Loss = loss(y,G,L,T)
            print(p,r,s," No. of iter: ", iter_no)
            # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
            # print("Final loss: ", Loss)
            return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        elif (stop_method == 'Est') and (np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro') < stop_thres):
            Loss = loss(y,G,L,T)
            print("No. of iter: ", iter_no)
            print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
            print("Final loss: ", Loss)
            return A,lmbd,gamma,theta,G,Loss,flag_maxiter
        # elif ((stop_method == 'Both') & (pre_loss-Loss < 0)):
            # print("No. of iter: ", iter_no)
            # print("Final est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
            # print("Warning loss: ", Loss)
            # print("Final grad: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
            # return A,lmbd,gamma,theta,G,Loss,flag_maxiter 
        # if iter_no%1 == 0:
        #     print("est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
            # print("lmbd's diff: ", np.linalg.norm(lmbd-pre_lmbd, ord=2))
            # print("gamma's diff: ", np.linalg.norm(gamma-pre_gamma, ord=2))
            # print("theta's diff: ", np.linalg.norm(theta-pre_theta, ord=2))
            # print("G's diff: ", np.linalg.norm(tensor_op.unfold(G-pre_G,1), ord='fro'))
            # print("G est err: ", np.linalg.norm(tensor_op.unfold(G-true_G[:,:,:T],1), ord='fro'))
            # print("A's diff: ", np.linalg.norm(tensor_op.unfold(A-pre_A,1), ord='fro'))
            # print(lmbd)
            # print("A est errr: ", np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:T],1), ord='fro'))
            # print(np.linalg.norm(np.concatenate([lmbd-lmbd_true,gamma-gamma_true,theta-theta_true]), ord=2))
            # print(gamma,theta)
            # print("loss: ", Loss)
        

    # L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    print(p,r,s," No. of iter: ", iter_no)
    flag_maxiter = 1
    Loss = loss(y,G,L,T)
    A = tl.tenalg.mode_dot(G,L,2)
    # print(gamma)
    return A,lmbd,gamma,theta,G,Loss,flag_maxiter 