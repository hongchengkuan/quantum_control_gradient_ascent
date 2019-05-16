function [ alpha ] = bisection_linesearch( x0, f0, trans_prob, d)
%BISECTION_LINESEARCH Linesearch, based on
% von Winckel, Borzi: Computational techniques for a quantum control problem with $H^{1}$-cost, Inverse Problems 2008
% Input:
% x0: starting point
% f0: Objectiv evaluated at x0
% J: the target functional; must have just one argument
% d: search direction
% alpha0: starting step length (ignored)
%
% Output:
% alpha: step length

alpha=1e-1; % step size alpha -> will be increased in the first step  startalpha 0.1 instead of 1e-6');
da=1e-4;

phi=@(a) -trans_prob(x0+a*d); % functional as a function of step size only
dphi=@(a) (phi(a+da)-phi(a-da))/(2*da); % derivative of functional for bisection

%% Increase step size till functional increases again
fl=f0;
fr=phi(alpha);
while fr<fl % as long as the functional is smaller to the right, increase step size
    alpha=(alpha+.3)*1.4;% some combination of linear and exponential growth
    fl=fr;
    fr=phi(alpha);
    if alpha > 1e8
        warning('Bisection linesearch: step length too large');
        break;
    end
end

al=eps; % left interval boundary
ar=alpha; % right interval boundary

%% Do the bisection: 
% Check if the sign of the derivative changes in the left half of the
% interval, then there is the minimum, so take this interval.
% Otherwise take right half of interval

dl=dphi(al);
if(dl>0)
    error('Bisection not in descent direction.');
end

am=(al+ar)/2; % midpoint of interval
dm=dphi(am);
while abs(ar-al)/am>1e-2  % as long as the interval is large; should be like 1e-6
    if dm*dl < 0 % is derivative changes sign in the left half
        ar=am; % the minimum is there
    else % otherwise take the right half of the interval
        al=am;
        dl=dm;
    end
    am=(al+ar)/2;
    dm=dphi(am);
end
alpha=am;
end