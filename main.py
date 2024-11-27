import GS2D
import numpy as np

def main():
    pass

    # Diffusion coefficients
    DA = 2*10**-5
    DB = 1*10**-5

    # define feeding/killing rates
    f = 0.032
    k = 0.061

    # grid size
    N = 100 # 128

    # time and spatial step
    delta_t = 1.0 
    dx = 1/N  #1.0 / N

    # intialization
    U, V = GS2D.IC(N)
    # print("A: ", A, " B: ", B)
    U_record = U.copy()[None,...]
    V_record = V.copy()[None,...]

    N_simulation_steps = 15000

    for step in range(N_simulation_steps):
        U, V = GS2D.update_rk4(U, V, DA, DB, f, k, delta_t, dx)
        
        if step%5 ==0:
            U_record = np.concatenate((U_record, U[None,...]), axis=0)
            V_record = np.concatenate((V_record, V[None,...]), axis=0)
        
    UV = np.concatenate((U_record[None,...], V_record[None,...]), axis=0)

    # plotting & saving
    output = np.transpose(UV, [1, 0, 2, 3])
    fig_save_path = f'./k={k}_f={f}/'
    for i in range(21):
        GS2D.postProcess(output, N, 0, N*dx, 0, N*dx, num=150*i, batch=1,save_path=fig_save_path)