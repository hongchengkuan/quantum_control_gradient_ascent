import quantum_control_utils as utils
import numpy as np
import time
import matplotlib.pyplot as plt
qcm = utils.QuantumControlMorse((3,), (5,))
Ef = np.zeros(qcm.N)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
iter = 0
while True:
    time1 = time.time()
    prob, psi, grad_vec = qcm.transition_prob(Ef, 3)
    time_prob = time.time() - time1

    print('grad takes %f s' % time_prob)
    print('probability = %.4e' % prob)
    if prob > .99:
        break

    alpha = qcm.bisection_line_search(Ef, prob, grad_vec)
    print('alpha = %f' % alpha)
    Ef += alpha * grad_vec

    iter += 1
    time_iter = time.time() - time1
    print('iteration %d takes %f s' % (iter, time_iter))

np.save('/home/rlair/work/chong009/quantum_control/numericalmethod/gradientdescent/quantum_control_gradient_ascent/3to5longer/Ef', Ef)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
ax1.plot(qcm.x, qcm.potential)
ax1.set_title(r'potential')
ax2.plot(qcm.x, np.abs(qcm.psi_target))
ax2.set_title(r'$\Psi_1$')
ax3.plot(qcm.x, np.abs(psi))
ax3.set_title(r'$\Psi_T$')
ax4.plot(qcm.t, Ef)
ax4.set_title(r'electric field')
plt.show()
