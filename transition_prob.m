function [prob,grad_vector,psi] = transition_prob(Ef, psi0,psi_target, potential, m, x, tau)
    dx = x(2)-x(1);
    L = length(potential);
    N = length(Ef);
    mu = 3.088*x.*exp(-x/.6);
    ud2 = 1i*tau/(4*12*m*dx^2)*ones(L,1);
    ud1 = -16*1i*tau/(4*12*m*dx^2)*ones(L,1);
    
    psi_all = cell(N,1);
    psi_all{1}=psi0;

    for j = 2:N
        psi_pre = psi_all{j-1};
        r = psi_pre-1i*tau/2*(-1/(2*12*m*dx^2)*(-[psi_pre(3:end);0;0]+16*[psi_pre(2:end);0]...
            -30*psi_pre+16*[0;psi_pre(1:end-1)]-[0;0;psi_pre(1:end-2)])+(potential - Ef(j-1)*mu).*psi_pre);
        
        dia_U = 1+30*1i*tau/(4*12*m*dx^2)+1i*tau/2*(potential - Ef(j)*mu);
        U = spdiags([ud2 ud1 dia_U ud1 ud2],-2:2,L,L);
        psi_all{j} = U\r;
    end
    psi = psi_all{N};
    
    grad_Ef=cell(1,N);
    for j = N:-1:1
        if j == N
            dia_U = 1+30*1i*tau/(4*12*m*dx^2)+1i*tau/2*(potential - Ef(j)*mu);
            U = spdiags([ud2 ud1 dia_U ud1 ud2],-2:2,L,L);
            
            grad_Ef{j} = 1i*tau/2* (U\(mu.*psi_all{N}));
        elseif j == N-1
            U_PRE = U;
            dia_U = 1+30*1i*tau/(4*12*m*dx^2)+1i*tau/2*(potential - Ef(j)*mu);
            U = spdiags([ud2 ud1 dia_U ud1 ud2],-2:2,L,L);
            dia_IH = 1-30*1i*tau/(4*12*m*dx^2)-1i*tau/2*(potential - Ef(j)*mu);
            IH = spdiags([-ud2 -ud1 dia_IH -ud1 -ud2], -2:2,L,L);
            AUIH = full(U_PRE\IH);
            temp = mu.*psi_all{N-1};
            
            grad_Ef{j} = 1i*tau/2*(AUIH * (U \ temp) +...
                U_PRE \ temp);
        elseif j>= 2
            U_PRE = U;
            dia_U = 1+30*1i*tau/(4*12*m*dx^2)+1i*tau/2*(potential - Ef(j)*mu);
            U = spdiags([ud2 ud1 dia_U ud1 ud2],-2:2,L,L);
            dia_IH = 1-30*1i*tau/(4*12*m*dx^2)-1i*tau/2*(potential - Ef(j)*mu);
            IH = spdiags([-ud2 -ud1 dia_IH -ud1 -ud2], -2:2,L,L);
            AUIH_PRE = AUIH;
            AUIH = AUIH * full(U_PRE\IH);
            temp = mu.*psi_all{j};

            grad_Ef{j} = 1i*tau/2 * (AUIH * (U\temp) + AUIH_PRE * (U_PRE\temp));
        else
            grad_Ef{j} = 1i*tau/2*AUIH * (U\(mu.*psi_all{j}));
        end
    end
    prob = abs(sum((psi_target.*psi*(x(2)-x(1)))))^2;
    grad_vector = 2*real((psi_target'*psi)*conj(psi_target'*[grad_Ef{:}]))*dx^2;

end

