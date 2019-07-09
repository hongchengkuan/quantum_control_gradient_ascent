import quantum_control_utils as utils
import numpy as np
import time
import matplotlib.pyplot as plt
qcm = utils.QuantumControlMorse((3,), (5,))
Ef = np.load('/home/rlair/work/chong009/quantum_control/numericalmethod/gradientdescent/quantum_control_gradient_ascent/3to5longer/Ef.npy')
prob, psi = qcm.transition_prob(Ef, 2)
print(prob)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
ax1.plot(qcm.x, qcm.potential)
ax1.set_title('potential')
ax2.plot(qcm.x, np.abs(qcm.psi_target))
ax2.set_title(r'$\Psi_1$')
ax3.plot(qcm.x, np.abs(psi))
ax3.set_title(r'$\Psi_T$')
ax4.plot(qcm.t, Ef, linewidth=0.1)
ax4.set_title('electric field')
plt.tight_layout()
fig2,ax5 = plt.subplots()
ax5.plot(qcm.t, Ef, linewidth=0.1)
ax5.set_title('optimized electric field')
ax5.set_ylabel(r'$\epsilon(t)$ (a.u.)')
ax5.set_xlabel(r'$t$ (a.u.)')

plt.tight_layout()
plt.show()
