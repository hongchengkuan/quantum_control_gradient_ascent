function [psi,Ef] = cga(trans_prob,x0)
% based on
% Sprengal, Ciaramella and Borzi: A COKOSNUT code for the control of the
% time-dependent Kohn-Sham model
iter = 0;
Ef = x0;
while 1
    tic;
    [prob, psi,grad] = trans_prob(Ef);
    
    t1=toc;
    iter
    prob
    fprintf('grad evaluation takes %f seconds\n',t1);
    if prob > .99
        fprintf('optimization done');
        save('./Ef.mat','Ef')
        break;
    end
    
    if iter == 0
        Ef = x0 + grad;
        d = grad;
    else

        g_norm = norm(grad);
        if abs(grad'*grad_pre)/g_norm^2 >= 0.5
            beta = 0;
        else
            y = -(grad-grad_pre);
            sigma = y-2*d*norm(y)^2/(d'*y);
            beta = -sigma'*grad/(d'*y);
            
            eta = -1/(norm(d)*min(0.01,norm(grad_pre)));
            beta = max(beta,eta);
        end
        d = grad+beta*d;
        alpha = bisection_linesearch(Ef,-prob,trans_prob,d);
        Ef = Ef + alpha*d;
    end
    grad_pre = grad;
    iter = iter + 1;
    
    t2=toc;
    fprintf('iteration takes %f seconds\n',t2);
end
end


