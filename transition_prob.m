function [prob,grad_vector,psi] = transition_prob(Ef, psi0,psi_target, potential, m, x, tau)
    dx = x(2)-x(1);
    L = length(potential);
    N = length(Ef);
    mu = 3.088*x.*exp(-x/.6);
    ud2 = 1i*tau/(4*12*m*dx^2)*ones(L,1);
    ud1 = -16*1i*tau/(4*12*m*dx^2)*ones(L,1);
    
    psi_all = cell(N,1);
    psi_all{1}=psi0;
    U = cell(N-1,1);
    IH = cell(N-1,1);

    for j = 2:N
        psi_pre = psi_all{j-1};
        r = psi_pre-1i*tau/2*(-1/(2*12*m*dx^2)*(-[psi_pre(3:end);0;0]+16*[psi_pre(2:end);0]...
            -30*psi_pre+16*[0;psi_pre(1:end-1)]-[0;0;psi_pre(1:end-2)])+(potential - Ef(j-1)*mu).*psi_pre);
        
        dia_IH = 1-30*1i*tau/(4*12*m*dx^2)-1i*tau/2*(potential - Ef(j-1)*mu);
        IH{j-1} = spdiags([-ud2 -ud1 dia_IH -ud1 -ud2], -2:2,L,L);
        
        dia = 1+30*1i*tau/(4*12*m*dx^2)+1i*tau/2*(potential - Ef(j)*mu);
        U{j-1} = spdiags([ud2 ud1 dia ud1 ud2],-2:2,L,L);
        psi_all{j} = U{j-1}\r;
    end
    psi = psi_all{N};
    
    grad_Ef=cell(1,N);
    for j = N:-1:1
        if j == N
            grad_Ef{j} = 1i*tau/2* (U{j-1}\(mu.*psi_all{N}));
        elseif j == N-1
            AUIH = full(U{j}\IH{j});
            grad_Ef{j} = 1i*tau/2*(AUIH * (U{j-1} \ (mu.*psi_all{N-1})) +...
                U{j} \ (mu.*psi_all{N-1}));
        elseif j>= 2
            AUIH_PRE = AUIH;
            AUIH = AUIH * full(U{j}\IH{j});
            temp = mu.*psi_all{j};
            grad_Ef{j} = 1i*tau/2 * (AUIH * (U{j-1}\temp) + AUIH_PRE * (U{j}\temp));
        else
            grad_Ef{j} = 1i*tau/2*AUIH * (U{j}\(mu.*psi_all{j}));
        end
    end
    prob = abs(sum((psi_target.*psi*(x(2)-x(1)))))^2;
    grad_vector = 2*real((psi_target'*psi)*conj(psi_target'*[grad_Ef{:}]))*dx^2;

end

