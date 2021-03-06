clc;clear;
x_min = 1.35;
x_max = 2.85;
x = linspace(x_min,x_max,128)';
t_min = 0;
t_max = 30000;
t = linspace(t_min,t_max,100000)';
m = 1728.468338;
mu = 3.088*x.*exp(-x/.6);

%The time intervel
tau = t(2) - t(1);
%The distance interval
dx = x(2) - x(1);

potential = V_Morse(x,.1994,1.189,1.821);
[states,~] = TI_solve(potential, m, dx, 10);
states = states/sqrt(dx);

psi_target = states(:,2);
psi_init = states(:,1);

% This function calculates the transition probability, the gradient w.r.t. 
% the electric field at each time step, and the wave function at the end
trans_prob = @(Ef) transition_prob(Ef,psi_init,psi_target,potential, m, ...
    mu,dx,tau);

%% optimization begins
iter=0;
Ef = zeros(length(t),1);
% sga uses steepest gradient, cga uses conjugate gradient
[psi,Ef]=sga(trans_prob,Ef);

%% plot the functions
subplot(4,1,1)
plot(x,potential)
title('potential')
subplot(4,1,2)
plot(x, abs(psi_target).^2)
title('|\psi_{1}(x)|^{2}')
subplot(4,1,3)
plot(x,abs(psi).^2);
title('|\psi_T(x)|^2');
subplot(4,1,4)
plot(t,Ef);
title('electric field');

