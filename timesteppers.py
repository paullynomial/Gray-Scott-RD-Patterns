import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque
from farray import axslice, apply_matrix

class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt = dt
        self.t += dt
        self.iter += 1

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.F = eq_set.F
        if hasattr(eq_set, 'BC'):
            self.BC = eq_set.BC
        else:
            self.BC = None

    def step(self, dt):
        super().step(dt)
        if self.BC:
            self.BC(self.X)
            self.X.scatter()


class ImplicitTimestepper(Timestepper):

    def __init__(self, eq_set, axis):
        super().__init__()
        self.axis = axis
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        N = len(self.X.data)
        self.I = sparse.eye(N, N)

    def _LUsolve(self, data):
        if self.axis == 0:
            return self.LU.solve(data)
        elif self.axis == len(data.shape)-1:
            return self.LU.solve(data.T).T
        else:
            raise ValueError("Can only do implicit timestepping on first or last axis")


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.X.data + dt*self.F(self.X)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, eq_set):
        super().__init__(eq_set)
        N = len(X.data)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.X.data + dt*self.F(self.X)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.X_old = np.copy(self.X.data)
            return self.X + dt*self.F(self.X)
        else:
            X_temp = self.X_old + 2*dt*self.F(self.X)
            self.X_old = np.copy(self.X)
            return X_temp


class Multistage(ExplicitTimestepper):

    def __init__(self, eq_set, stages, a, b):
        super().__init__(eq_set)
        self.stages = stages
        self.a = a
        self.b = b

        self.X_list = []
        self.K_list = []
        for i in range(self.stages):
            self.X_list.append(StateVector([np.copy(var) for var in self.X.variables]))
            self.K_list.append(np.copy(self.X.data))

    def _step(self, dt):
        X = self.X
        X_list = self.X_list
        K_list = self.K_list
        stages = self.stages

        np.copyto(X_list[0].data, X.data)
        for i in range(1, stages):
            K_list[i-1] = self.F(X_list[i-1])

            np.copyto(X_list[i].data, X.data)
            # this loop is slow -- should make K_list a 2D array
            for j in range(i):
                X_list[i].data += self.a[i, j]*dt*K_list[j]
            if self.BC:
                self.BC(X_list[i])

        K_list[-1] = self.F(X_list[-1])

        # this loop is slow -- should make K_list a 2D array
        for i in range(stages):
            X.data += self.b[i]*dt*K_list[i]

        return X.data


def RK22(eq_set):
    a = np.array([[  0,   0],
                  [1/2,   0]])
    b = np.array([0, 1])
    return Multistage(eq_set, 2, a, b)


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.f_list = deque()
        for i in range(self.steps):
            self.f_list.append(np.copy(X.data))

    def _step(self, dt):
        f_list = self.f_list
        f_list.rotate()
        f_list[0] = self.F(self.X)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter+1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.X.data += self.dt*coeff*self.f_list[i].data
        return self.X.data

    def _coeffs(self, num):
        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i)**(j-1)/factorial(j-1)

        b = (-1)**(j+1)/factorial(j)

        a = np.linalg.solve(S, b)
        return a


class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self._LUsolve(self.X.data)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt/2*self.L
            self.RHS = self.M - dt/2*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self._LUsolve(apply_matrix(self.RHS, self.X.data, self.axis))


class BackwardDifferentiationFormula(ImplicitTimestepper):

    def __init__(self, u, L, steps):
        pass

    def _step(self, dt):
        pass


class StateVector:

    def __init__(self, variables, axis=0):
        self.axis = axis
        var0 = variables[0]
        shape = list(var0.shape)
        self.N = shape[axis]
        shape[axis] *= len(variables)
        self.shape = tuple(shape)
        self.data = np.zeros(shape)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[axslice(self.axis, i*self.N, (i+1)*self.N)], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[axslice(self.axis, i*self.N, (i+1)*self.N)])


class IMEXTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F

    def step(self, dt):
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt = dt
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        
        RHS = self.M @ self.X.data + dt*self.F(self.X)
        return self.LU.solve(RHS)


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt or self.iter == 1:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)


#hw5 starts here
class BDFExtrapolate(IMEXTimestepper):
    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.previous_X = [np.zeros_like(self.X.data) for _ in range(self.steps + 1)]
        self.previous_FX = [np.zeros_like(self.F(self.X)) for _ in range(self.steps)]
        self.previous_X[0] = self.X.data.copy()
        self.previous_FX[0] = self.F(self.X).copy()

        self.dim = steps + 1

        self.dt_end = None
        self.LU_init = None
        self.LU_bdf = None

    def get_coeff(self, dt):

        A_1 = np.array([[(i * -dt) ** j / math.factorial(j) for j in range(self.dim)] for i in range(self.dim)])
        a = np.linalg.solve(A_1.T, [0, 1] + [0] * (self.steps - 1))

        A_2 = np.array([[((i + 1) * -dt) ** j / math.factorial(j) for j in range(self.steps)] for i in range(self.steps)])
        b = np.linalg.solve(A_2.T, [1] + [0] * (self.steps - 1))
        return a, b

    def get_LHS(self, dt, is_initial):
        if is_initial:
            LHS = self.M + dt * self.L
        else:
            a0 = self.get_coeff(dt)[0][0]
            LHS = a0 * self.M + self.L
        
        LU = spla.splu(LHS.tocsc())
        
        if is_initial:
            self.LU_init = LU
            self.dt_end = dt
        else:
            self.LU_bdf = LU

    def get_RHS(self, dt):
        if self.iter < self.steps:
            return self.M @ self.X.data + dt * self.F(self.X)
        else:
            a, b = self.get_coeff(dt)
            return -self.M @ (a[1:] @ np.array(self.previous_X[:-1])) + b @ np.array(self.previous_FX)
    
    def chose_LHS(self, dt):
        if self.iter < self.steps:
            if not self.LU_init or self.dt_end != dt:
                self.get_LHS(dt, is_initial=True)
            return self.LU_init
        else:
            if not self.LU_bdf:
                self.get_LHS(dt, is_initial=False)
            return self.LU_bdf
        
    def _step(self, dt):
        #update
        self.previous_FX = np.roll(self.previous_FX, 1, axis=0)
        self.previous_FX[0] = self.F(self.X)
        self.previous_X = np.roll(self.previous_X, 1, axis=0)
        self.previous_X[0] = self.X.data

        LHS = self.chose_LHS(dt)
        RHS = self.get_RHS(dt)
        return LHS.solve(RHS)


class FullyImplicitTimestepper(Timestepper):

    def __init__(self, eq_set, tol=1e-5):
        super().__init__()
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.tol = tol
        self.J = eq_set.J

    def step(self, dt, guess=None):
        self.X.gather()
        self.X.data = self._step(dt, guess)
        self.X.scatter()
        self.t += dt
        self.iter += 1


class BackwardEulerFI(FullyImplicitTimestepper):

    def _step(self, dt, guess):
        if dt != self.dt:
            self.LHS_matrix = self.M + dt*self.L
            self.dt = dt

        RHS = self.M @ self.X.data
        if not (guess is None):
            self.X.data[:] = guess
        F = self.F(self.X)
        LHS = self.LHS_matrix @ self.X.data - dt * F
        residual = LHS - RHS
        i_loop = 0
        while np.max(np.abs(residual)) > self.tol:
            jac = self.M + dt*self.L - dt*self.J(self.X)
            dX = spla.spsolve(jac, -residual)
            self.X.data += dX
            F = self.F(self.X)
            LHS = self.LHS_matrix @ self.X.data - dt * F
            residual = LHS - RHS
            i_loop += 1
            if i_loop > 20:
                print('error: reached more than 20 iterations')
                break
        return self.X.data


class CrankNicolsonFI(FullyImplicitTimestepper):

    def _step(self, dt, guess):
        L_matrix = self.M + dt / 2 * self.L
        R_matrix = self.M - dt / 2 * self.L

        F = self.F(self.X)
        RHS = R_matrix @ self.X.data + dt / 2 * F
        if not (guess is None):
            self.X.data[:] = guess
        F = self.F(self.X)
        LHS = L_matrix @ self.X.data - dt / 2 * F
        residual = LHS - RHS
        i_loop = 0
        while np.abs(residual).max() > self.tol:
            jac = L_matrix - dt / 2 * self.J(self.X)
            dX = spla.spsolve(jac, -residual)
            self.X.data += dX
            F = self.F(self.X)
            LHS = L_matrix @ self.X.data - dt / 2 * F
            residual = LHS - RHS
            i_loop += 1
            if i_loop > 20:
                print("error: reached more than 20 iterations")
                break
        return self.X.data

