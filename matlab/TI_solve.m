function [psi,E] = TI_solve(potential, m, dx, k)
    %%The initial Halmiltonian for 4th order finite difference
    %ud2 = 1/(2*12*m*dx^2)*ones(L,1);
    %ud1 = -16/(2*12*m*dx^2)*ones(L,1);
    %dia = 30/(2*12*m*dx^2)+potential;
    %H = spdiags([ud2 ud1 dia ud1 ud2],-2:2,L,L);
    %[psi,E] = eigs(H,k,'sa');
    
    %%The initial Halmiltonian for 2nd order finite difference
    L = length(potential);
    e = ones(L,1)/(2*m*dx^2);
    H = spdiags([-e 2*e+potential -e],-1:1,L,L);
    [psi,E] = eigs(H,k,'sa');
    
    %The sign here is just used to make sure the first element of each wave
    %function is positive to align the sign in MATLAB and Python
    sign = 1*(psi(1,:)>0)-1*(psi(1,:)<0);
    psi = sign .* psi;
end

