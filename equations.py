from timesteppers import StateVector, CrankNicolson, RK22
from scipy import sparse
import finite
import numpy as np


#hw7 starts here
class Diffusionx:

    def __init__(self, c, D, d2x):
        self.X = StateVector([c], axis=0)
        N = c.shape[0]
        M = sparse.eye(N, N)
        L = -D*d2x.matrix
        # apply boundary conditions
        M = M.tocsr()
        M[0,:] = 0
        M[-1,:] = 0
        M.eliminate_zeros()
        L[0,:] = 0
        L[0, 0] = 1
        BC_vector = np.zeros(N)
        BC_vector[-3] = -1/(2*d2x.dx)
        BC_vector[-2] = 2/(d2x.dx)
        BC_vector[-1] = -3/(2*d2x.dx)
        L[-1, :] = BC_vector
        L.eliminate_zeros()
        self.M = M
        self.L = L

class Diffusiony:

    def __init__(self, c, D, d2y):
        self.X = StateVector([c], axis=1)
        N = c.shape[1]
        self.M = sparse.eye(N, N)
        self.L = -D*d2y.matrix

class DiffusionBC:

    def __init__(self, c, D, spatial_order, domain):
        self.t = 0
        self.iter = 0

        self.X = StateVector([c])
        grid_x, grid_y = domain.grids
        self.dx2 = finite.DifferenceUniformGrid(2, spatial_order, grid_x, 0)
        self.dy2 = finite.DifferenceUniformGrid(2, spatial_order, grid_y, 1)
        self.Diffusionx = Diffusionx(c, D, self.dx2)
        self.Diffusiony = Diffusiony(c, D, self.dy2)
        self.diffusion_x = CrankNicolson(self.Diffusionx, axis=0)
        self.diffusion_y = CrankNicolson(self.Diffusiony, axis=1)

    def step(self, dt):
        self.diffusion_x.step(dt/2)
        self.diffusion_y.step(dt)
        self.diffusion_x.step(dt/2)
        self.t += dt
        self.iter += 1

class Wave2DBC:

    def __init__(self, u, v, p, spatial_order, domain):
        self.X = StateVector([u, v, p])
        self.N = len(u)
        self.order = spatial_order // 2

        grid_x, grid_y = domain.grids
        self.dx = finite.DifferenceUniformGrid(1, spatial_order, grid_x, axis=0)
        self.dy = finite.DifferenceUniformGrid(1, spatial_order, grid_y, axis=1)

        self.BC = self.BC
        self.F = self.F

    def F(self, X):
        X.scatter()
        u, v, p = X.variables
        du = self.dx @ p
        dv = self.dy @ p
        dp = self.dx @ u + self.dy @ v
        return np.concatenate((-du, -dv, -dp))

    def BC(self, X):
        X.data[0] = np.zeros(self.N)
        X.data[self.N - 1] = np.zeros(self.N)
#hw7 ends here


#hw6 starts here
class ReactionDiffusion_x:
    
    def __init__(self, c, D, dx2):
        self.X = StateVector([c])
        N = len(c)
        self.M = sparse.eye(N, N)
        self.L = -D*dx2.matrix

class ReactionDiffusion_y:
    
    def __init__(self, c, D, dy2):
        self.X = StateVector([c])
        N = len(c)
        self.M = sparse.eye(N, N)
        self.L = -D*dy2.matrix

class ReactionDiffusion2D:

    def __init__(self, c, D, dx2, dy2):
        # for rk22
        self.X = StateVector([c])
        self.F = lambda X: X.data * (1 - X.data)
        # time stepper for reaction
        self.ts_reaction = RK22(self)
        # for diffusion
        self.eq_setx = ReactionDiffusion_x(c, D, dx2)
        self.eq_sety = ReactionDiffusion_y(c, D, dy2)
        self.diffusion_x = CrankNicolson(self.eq_setx, axis=0)
        self.diffusion_y = CrankNicolson(self.eq_sety, axis=1)
        # iter count
        self.t = 0
        self.iter = 0
    
    def step(self, dt):
        # reaction 1/2 step
        self.ts_reaction.step(dt/2)
        # diffusion  1 step
        self.diffusion_x.step(dt/2)
        self.diffusion_y.step(dt)
        self.diffusion_x.step(dt/2)
        # reaction 1/2 step
        self.ts_reaction.step(dt/2)
        
        self.t += dt
        self.iter += 1


class VB_Diffusion_x:
    def __init__(self, u, v, nu, dx2):
        self.X = StateVector([u, v], axis=0)
        N = len(u)
        self.M = sparse.eye(N*2, N*2) 
        self.L = -nu*sparse.block_diag((dx2.matrix, dx2.matrix))

class VB_Diffusion_y:
    def __init__(self, u, v, nu, dy2):
        self.X = StateVector([u, v], axis=1)
        M = len(u)
        self.M = sparse.eye(M*2, M*2)
        self.L = -nu*sparse.block_diag((dy2.matrix, dy2.matrix))

class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        self.X = StateVector([u, v])
        # spatial derivative operators
        grid_x, grid_y = domain.grids
        self.dx = finite.DifferenceUniformGrid(1, spatial_order, grid_x, 0)
        self.dy = finite.DifferenceUniformGrid(1, spatial_order, grid_y, 1)
        self.dx2 = finite.DifferenceUniformGrid(2, spatial_order, grid_x, 0)
        self.dy2 = finite.DifferenceUniformGrid(2, spatial_order, grid_y, 1)
        # advection F
        def f(X):
            X.scatter()
            u,v = X.variables
            return -np.concatenate((u * (self.dx @ u) + v * (self.dy @ u), u * (self.dx @ v)+v* (self.dy @ v)), axis=0)
        self.F = f
        self.u, self.v = self.X.variables
        self.ts_advection = RK22(self)          
        # diffusion M & L
        self.eq_setx = VB_Diffusion_x(self.u, self.v, nu, self.dx2)
        self.eq_sety = VB_Diffusion_y(self.u, self.v, nu, self.dy2)
        self.ts_diffusion_x = CrankNicolson(self.eq_setx, axis=0)
        self.ts_diffusion_y = CrankNicolson(self.eq_sety, axis=1)
        
        self.t = 0
        self.iter = 0

    def step(self, dt):
        # advection half step
        self.ts_advection.step(dt/2)
        
        # diffusion full step
        self.ts_diffusion_x.step(dt/2)
        self.ts_diffusion_y.step(dt)
        self.ts_diffusion_x.step(dt/2)
        
        # advection half step
        self.ts_advection.step(dt/2)
        
        #update
        self.t += dt
        self.iter += 1
#hw6 ends here


class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        self.F = lambda X: 0*X.data


class SoundWave:

    def __init__(self, u, p, d, rho0, gammap0):
        self.X = StateVector([u, p])
        N = len(u)
        
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = rho0 * I if np.isscalar(rho0) else sparse.diags(rho0)
        M01 = Z
        M10 = Z
        M11 = I

        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = d.matrix
        L10 = gammap0 * d.matrix if np.isscalar(gammap0) else sparse.diags(gammap0) @ d.matrix
        L11 = Z

        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        self.F = lambda X: 0*X.data


class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        N= len(c)
        self.M = sparse.eye(N, N)
        self.L = -D * d2.matrix
        self.F = lambda X: X.data * (c_target - X.data)

class ReactionDiffusionFI:
    
    def __init__(self, c, D, spatial_order, grid):
        self.X = StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -D*d2.matrix

        def F(X):
            return X.data*(1-X.data)
        self.F = F
        
        def J(X):
            c_matrix = sparse.diags(X.data)
            return sparse.eye(self.N) - 2*c_matrix
        
        self.J = J


#hw8 starts here
class BurgersFI:
    def __init__(self, u, nu, spatial_order, grid):
        N = len(u)
        if isinstance(nu, (int, float)):
            nu = np.full((N,), nu)
        if isinstance(nu, np.ndarray):
            nu = sparse.diags(nu)
        d2x = finite.DifferenceUniformGrid(2, spatial_order, grid, 0)
        dx = finite.DifferenceUniformGrid(1, spatial_order, grid, 0)
        self.X = StateVector([u])
        self.M = sparse.eye(N)
        self.L = -nu @ d2x.matrix

        def F(X):
            return -np.multiply(X.data, dx @ X.data)
        self.F = F

        def J(X):
            return -(sparse.diags(X.data) @ dx.matrix + sparse.diags(dx @ X.data))
        self.J = J


class ReactionTwoSpeciesDiffusion:
    
    def __init__(self, X, D, r, spatial_order, grid):
        d2x = finite.DifferenceUniformGrid(2, spatial_order, grid, 0)

        N, *_ = X.variables[0].shape

        if isinstance(D, (int, float)):
            D = np.full((N,), D, dtype=np.float64)
        if isinstance(D, np.ndarray):
            D = sparse.diags(np.concatenate((D, D)))

        if isinstance(r, (int, float)):
            r = np.full((N,), r, dtype=np.float64)
        if isinstance(r, np.ndarray):
            r = sparse.diags(r)

        self.X = X 
        self.M = sparse.eye(2 * N)
        self.L = -D @ sparse.bmat(((d2x.matrix, sparse.csr_matrix((N, N))), (sparse.csr_matrix((N, N)), d2x.matrix),))

        def F(X):
            X.scatter() 
            c1, c2 = X.variables
            return np.concatenate((c1 * (1 - c1 - c2), r @ (c2 * (c1 - c2))))

        self.F = F

        def J(X):
            X.scatter()
            c1, c2 = X.variables
            return sparse.bmat(((sparse.diags(1 - 2 * c1 - c2), sparse.diags(-c1)), (r @ sparse.diags(c2), r @ sparse.diags(c1 - 2 * c2)),))

        self.J = J
#hw8 ends here