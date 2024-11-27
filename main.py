import GS2D
import numpy as np
import os

def main():

    '''
    By changing the feeding and killing rates, we can generate a variety of patterns.
    Here are some references:
    bean like:
    f = 0.016
    k = 0.05
    or 
    f = 0.018
    k = 0.045

    maze:
    f = 0.0248
    k = 0.055
    or
    f = 0.06
    k = 0.063
    or
    f = 0.04
    k = 0.06

    dots:
    f = 0.03
    k = 0.055
    or
    f = 0.014
    k = 0.055
    or
    f = 0.024
    k = 0.061
    or
    f = 0.04
    k = 0.065

    dots and stripes:
    f = 0.039
    k = 0.064

    stripes:
    f = 0.054
    k = 0.065

    '''

    # Diffusion coefficients
    DA = 2*10**-5
    DB = 1*10**-5

    # feeding/killing rates
    f = 0.016
    k = 0.05

    # grid size
    N = 100 # 128

    # time and spatial step
    delta_t = 1.0 
    dx = 1/N  #1.0 / N

    # intialization
    U, V = GS2D.IC(N)
    U_record = U.copy()[None,...]
    V_record = V.copy()[None,...]

    N_simulation_steps = 5000

    saving_step = 5

    for step in range(N_simulation_steps):
        U, V = GS2D.evolve_rk4(U, V, DA, DB, f, k, delta_t, dx)
        
        if step%saving_step ==0:
            U_record = np.concatenate((U_record, U[None,...]), axis=0)
            V_record = np.concatenate((V_record, V[None,...]), axis=0)
        
    UV = np.concatenate((U_record[None,...], V_record[None,...]), axis=0)
    output = np.transpose(UV, [1, 0, 2, 3])

    # plotting & saving
    plotting_interval = 10
    fig_save_path = f'./k={k}_f={f}/'
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    for i in range(N_simulation_steps//(plotting_interval*saving_step)+1):
        GS2D.plotq(output, N, 0, N*dx, 0, N*dx, num=plotting_interval*i, save_path=fig_save_path)

    # make a gif
    GS2D.make_gif(fig_save_path)

main()
