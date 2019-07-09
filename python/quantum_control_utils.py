import numpy as np
import scipy.sparse as spa
from scipy.linalg import solve_banded
import scipy.sparse.linalg as sla


class QuantumControlMorse:
    def __init__(self, init, target):
        self.x_min = 1.35
        self.x_max = 2.85
        self.x = np.linspace(self.x_min, self.x_max, 128)
        self.t_min = 0
        # self.t_max = 30000
        # self.t = np.linspace(self.t_min, self.t_max, 100000)
        self.t_max = 250000
        self.t = np.linspace(self.t_min, self.t_max, 833333)

        self.m = 1728.468338
        self.mu = 3.088 * self.x * np.exp(-self.x / .6)

        self.tau = self.t[2] - self.t[1]

        self.dx = self.x[2] - self.x[1]

        self.L = len(self.x)
        self.N = len(self.t)

        self.potential = self.V_Morse(.1994, 1.189, 1.821)

        self.states = self.TI_solve(10)

        self.states = self.states / np.sqrt(self.dx)

        self.psi_target = np.zeros(self.L)
        self.psi_init = np.zeros(self.L)
        for index in target:
            if index > 0:
                self.psi_target += self.states[:, np.abs(index)]
            else:
                self.psi_target -= self.states[:, np.abs(index)]
        self.psi_target /= np.sqrt(len(target))
        for index in init:
            if index > 0:
                self.psi_init += self.states[:, np.abs(index)]
            else:
                self.psi_init -= self.states[:, np.abs(index)]
        self.psi_init /= np.sqrt(len(init))

    def transition_prob(self, Ef, nargout):
        m = self.m
        mu = self.mu
        potential = self.potential
        dx = self.dx
        tau = self.tau

        L = self.L
        N = self.N

        psi_init = self.psi_init
        psi_target = self.psi_target

        ud2 = 1j * tau / (4 * 12 * m * dx ** 2) * np.ones(L)
        ud1 = -16 * 1j * tau / (4 * 12 * m * dx ** 2) * np.ones(L)

        psi_all = np.zeros((N, L), dtype=np.complex)
        psi_all[0] = psi_init
        psi_pre = psi_init

        for j in range(1, N):
            r = psi_pre - 1j * tau / 2 * (-1 / (2 * 12 * m * dx ** 2) * (
                -np.concatenate((psi_pre[2:], np.array([0, 0]))) + 16 * np.concatenate((psi_pre[1:], np.array([0])))
                - 30 * psi_pre + 16 * np.concatenate((np.array([0]), psi_pre[0:-1])) - np.concatenate(
                    (np.array([0, 0]), psi_pre[0:-2]))) + (potential - Ef[j - 1] * mu) * psi_pre)
            dia = 1 + 30 * 1j * tau / (4 * 12 * m * dx ** 2) + 1j * tau / 2 * (potential - Ef[j] * mu)
            U_ab = np.array([ud2, ud1, dia, ud1, ud2])

            psi_pre = solve_banded((2, 2), U_ab, r)
            psi_all[j] = psi_pre

        psi = psi_pre
        prob = np.power(np.abs(np.sum(psi_target * psi * dx)), 2)

        if nargout == 2:
            return prob, psi
        else:
            grad_Ef = np.zeros((N, L), dtype=np.complex)

            for j in range(N - 1, -1, -1):
                if j == N - 1:
                    dia_U = 1 + 30 * 1j * tau / (4 * 12 * m * dx ** 2) + 1j * tau / 2 * (potential - Ef[j] * mu)
                    r = mu * psi_all[N-1]
                    U_ab = np.array([ud2, ud1, dia_U, ud1, ud2])
                    U_inv_r = solve_banded((2, 2), U_ab, r)
                    grad_Ef[j] = 1j * tau / 2 * U_inv_r

                elif j == N - 2:
                    U_PRE_ab = U_ab
                    dia_U = 1 + 30 * 1j * tau / (4 * 12 * m * dx ** 2) + 1j * tau / 2 * (potential - Ef[j] * mu)
                    U_ab = np.array([ud2, ud1, dia_U, ud1, ud2])
                    dia_IH = 1 - 30 * 1j * tau / (4 * 12 * m * dx ** 2) - 1j * tau / 2 * (potential - Ef[j] * mu)
                    IH = spa.spdiags(np.array([-ud2, -ud1, dia_IH, -ud1, -ud2]), np.arange(-2, 3), L, L).toarray()
                    AUIH = solve_banded((2, 2), U_PRE_ab, IH)
                    temp = mu * psi_all[N-2]
                    U_inv_temp = solve_banded((2, 2), U_ab, temp)
                    U_PRE_inv_temp = solve_banded((2, 2), U_PRE_ab, temp)

                    grad = 1j * tau / 2 * (np.matmul(AUIH, U_inv_temp) + U_PRE_inv_temp)
                    grad_Ef[j] = grad

                elif j >= 1:
                    U_PRE_ab = U_ab
                    dia_U = 1 + 30 * 1j * tau / (4 * 12 * m * dx ** 2) + 1j * tau / 2 * (potential - Ef[j] * mu)
                    U_ab = np.array([ud2, ud1, dia_U, ud1, ud2])
                    dia_IH = 1 - 30 * 1j * tau / (4 * 12 * m * dx ** 2) - 1j * tau / 2 * (potential - Ef[j] * mu)
                    IH = spa.spdiags(np.array([-ud2, -ud1, dia_IH, -ud1, -ud2]), np.arange(-2, 3), L, L).toarray()
                    AUIH_PRE = AUIH
                    AUIH = solve_banded((2, 2), U_PRE_ab, IH)
                    AUIH = np.matmul(AUIH_PRE, AUIH)

                    temp = mu * psi_all[j]
                    U_inv_temp = solve_banded((2, 2), U_ab, temp)
                    U_PRE_inv_temp = solve_banded((2, 2), U_PRE_ab, temp)

                    grad = 1j * tau / 2 * (np.matmul(AUIH, U_inv_temp) + np.matmul(AUIH_PRE, U_PRE_inv_temp))
                    grad_Ef[j] = grad

                else:
                    temp = mu * psi_all[j]
                    U_inv_temp = solve_banded((2, 2), U_ab, temp)
                    grad = 1j * tau / 2 * np.matmul(AUIH, U_inv_temp)
                    grad_Ef[j] = grad

            grad_vector = 2 * np.real(
                np.dot(psi_target, psi) * np.conjugate(np.matmul(grad_Ef, psi_target))) * dx ** 2
            return prob, psi, grad_vector.flatten()

    def V_Morse(self, D, a, x0):
        return D * (np.exp(-a * (self.x - x0)) - 1) ** 2 - D

    def TI_solve(self, k):
        L = self.L
        e = np.ones(L) / (2 * self.m * self.dx ** 2)
        H = spa.spdiags(np.array([-e, 2 * e + self.potential, -e]), np.array([-1, 0, 1]), L, L)
        E, psi = sla.eigsh(H, k, which='SA')

        sign = 1 * (psi[1, :] > 0) - 1 * (psi[1, :] < 0)
        psi = sign * psi
        return psi

    def bisection_line_search(self, x0, f0, d):
        alpha = 1e-1
        da = 1e-4

        def phi(a):
            prob, _ = self.transition_prob(x0+a*d, 2)
            return -prob

        def dphi(a):
            return (phi(a+da)-phi(a-da)) / (2*da)

        fl = f0
        fr = phi(alpha)

        while fr < fl:
            alpha = (alpha + .3) * 1.4
            fl = fr
            fr = phi(alpha)
            if alpha > 1e8:
                print('Bisection linesearch: step length too large')
                break

        al = np.finfo(float).eps
        ar = alpha

        dl = dphi(al)
        if dl > 0:
            raise NameError('Bisection not in descent direction.')

        am = (al + ar) / 2
        dm = dphi(am)
        while abs(ar - al) / am > 1e-2:
            if dm * dl < 0:
                ar = am
            else:
                al = am
                dl = dm
            am = (al + ar) / 2
            dm = dphi(am)

        alpha = am
        return alpha


class QuantumControlDoubleWell(QuantumControlMorse):
    def __init__(self, init, target, potential=None):
        self.x_min = -4
        self.x_max = 4
        self.x = np.linspace(self.x_min, self.x_max, 128)
        self.t_min = 0
        self.t_max = 2481
        self.t = np.linspace(self.t_min, self.t_max, 100000)

        self.m = 1
        self.mu = self.x

        self.tau = self.t[2] - self.t[1]

        self.dx = self.x[2] - self.x[1]

        self.L = len(self.x)
        self.N = len(self.t)

        if potential is None:
            self.potential = self.V_doublewell()
        else:
            self.potential = potential

        self.states = self.TI_solve(10)

        self.states = self.states / np.sqrt(self.dx)

        self.psi_target = np.zeros(self.L)
        self.psi_init = np.zeros(self.L)
        for index in target:
            if index > 0:
                self.psi_target += self.states[:, np.abs(index)]
            else:
                self.psi_target -= self.states[:, np.abs(index)]
        self.psi_target /= np.sqrt(len(target))
        for index in init:
            if index > 0:
                self.psi_init += self.states[:, np.abs(index)]
            else:
                self.psi_init -= self.states[:, np.abs(index)]
        self.psi_init /= np.sqrt(len(init))

    def V_doublewell(self):
        x = self.x
        return x ** 4 / 64 - x ** 2 / 4 + x ** 3 / 256
