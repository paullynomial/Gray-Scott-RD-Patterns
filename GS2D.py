import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as scio
from PIL import Image
import glob
import time
import re

## Laplacian Operator

# [[[[    0,   0, -1/12,   0,     0],
#    [    0,   0,   4/3,   0,     0],
#    [-1/12, 4/3,    -5, 4/3, -1/12],
#    [    0,   0,   4/3,   0,     0],
#    [    0,   0, -1/12,   0,     0]]]]

def apply_laplacian(mat, dx = 0.01):

    neigh_mat = -5*mat.copy()

    neighbors = [
                    ( 4/3,  (-1, 0) ),
                    ( 4/3,  ( 0,-1) ),
                    ( 4/3,  ( 0, 1) ),
                    ( 4/3,  ( 1, 0) ),
                    (-1/12,  (-2, 0)),
                    (-1/12,  (0, -2)),
                    (-1/12,  (0, 2)),
                    (-1/12,  (2, 0)),
                ]

    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0,1))

    return neigh_mat/dx**2

def update_rk4(U0, V0, DU, DV, f, k, delta_t, dx):

    ############# Stage 1 ##############
    # diffusion
    diff_U = DU * apply_laplacian(U0, dx)
    diff_V = DV * apply_laplacian(V0, dx)

    # reaction
    reaction = U0 * V0 ** 2
    diff_U -= reaction
    diff_V += reaction

    # feeding and killing
    diff_U += f * (1 - U0)
    diff_V -= (k + f) * V0

    K1_U = diff_U
    K1_V = diff_V

    ############# Stage 2 ##############
    U1 = U0 +  K1_U * delta_t/2.0
    V1 = V0 +  K1_V * delta_t/2.0

    # diffusion
    diff_U = DU * apply_laplacian(U1, dx)
    diff_V = DV * apply_laplacian(V1, dx)

    # reaction
    reaction = U1 * V1 ** 2
    diff_U -= reaction
    diff_V += reaction

    # feeding and killing
    diff_U += f * (1 - U1)
    diff_V -= (k + f) * V1

    K2_U = diff_U
    K2_V = diff_V

    ############# Stage 3 ##############

    U2 = U0 + K2_U * delta_t/2.0
    V2 = V0 + K2_V * delta_t/2.0

    # diffusion
    diff_U = DU * apply_laplacian(U2, dx)
    diff_V = DV * apply_laplacian(V2, dx)

    # reaction
    reaction = U2 * V2 ** 2
    diff_U -= reaction
    diff_V += reaction

    # feeding and killing
    diff_U += f * (1 - U2)
    diff_V -= (k + f) * V2

    K3_U = diff_U
    K3_V = diff_V

    ############# Stage 4 ##############
    U3 = U0 + K3_U * delta_t
    V3 = V0 + K3_V * delta_t

    # diffusion
    diff_U = DU * apply_laplacian(U3, dx)
    diff_V = DV * apply_laplacian(V3, dx)

    # reaction
    reaction = U3 * V3 ** 2
    diff_U -= reaction
    diff_V += reaction

    # feeding and killing
    diff_U += f * (1 - U3)
    diff_V -= (k + f) * V3

    K4_U = diff_U
    K4_V = diff_V

    # RK4
    U = U0 + delta_t*(K1_U+2*K2_U+2*K3_U+K4_U)/6.0
    V = V0 + delta_t*(K1_V+2*K2_V+2*K3_V+K4_V)/6.0

    return U, V

def IC(N):

    # initial homogeneous concentrations
    U = np.ones((N,N))
    V = np.zeros((N,N))

    # initial disturbance 
    N1, N2, N3 = N//4-4, N//2, 3*N//4
    r = int(N/10.0)
    
    # initial disturbance 1  
    # A[N1-r:N1+r, N1-r:N1+r] = 0.50
    # B[N1-r:N1+r, N1-r:N1+r] = 0.25

    # initial disturbance 2
    U[N1-r:N1+r, N3-r:N3+r] = 0.50
    V[N1-r:N1+r, N3-r:N3+r] = 0.25
#
#    # initial disturbance 3
#    A[N3-r:N3+r, N3-r:N3+r] = 0.50
#    B[N3-r:N3+r, N3-r:N3+r] = 0.25
#
#    # initial disturbance 4
#    A[N3-r:N3+r, N1-r:N1+r] = 0.50
#    B[N3-r:N3+r, N1-r:N1+r] = 0.25

    # initial disturbance 5
    U[N2-r:N2+r, N2-r:N2+r] = 0.50
    V[N2-r:N2+r, N2-r:N2+r] = 0.25
#
#    # initial disturbance 6
#    A[N2-r:N2+r, N3-r:N3+r] = 0.50
#    B[N2-r:N2+r, N3-r:N3+r] = 0.25

    return U, V

def postProcess(output, N, xmin, xmax, ymin, ymax, num, save_path):
    ''' num: Number of time step
    '''
    x = np.linspace(xmin, xmax, N+1)[:-1]
    y = np.linspace(ymin, ymax, N+1)[:-1]
    x_star, y_star = np.meshgrid(x, y)
    u_pred = output[num, 0, :, :]
    v_pred = output[num, 1, :, :]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0].scatter(x_star, y_star, c=u_pred, alpha=0.95, edgecolors='none', cmap='hot', marker='s', s=2)
    ax[0].axis('square')
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    cf.cmap.set_under('black')
    cf.cmap.set_over('white')
    ax[0].set_title('u')
    fig.colorbar(cf, ax=ax[0], extend='both')

    cf = ax[1].scatter(x_star, y_star, c=v_pred, alpha=0.95, edgecolors='none', cmap='hot', marker='s', s=2) #
    ax[1].axis('square')
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    cf.cmap.set_under('black')
    cf.cmap.set_over('white')
    ax[1].set_title('v')
    fig.colorbar(cf, ax=ax[1], extend='both')

    plt.savefig(save_path + 'uv_[t=%d].png'%(num))
    plt.close('all')


def make_gif(image_path):

    images = []

    # retrieving all images used to make gif
    file_list = sorted(glob.glob(image_path + '*.png'), key=lambda x: int(re.search(r'\[t=(\d+)\]', x).group(1)))
    
    # loop through all png files in the folder
    for filename in file_list: 
        im = Image.open(filename) 
        images.append(im) 

    # save as a gif   
    images[0].save(image_path + 'animation' + '.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)