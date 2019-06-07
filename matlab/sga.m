function [psi,Ef] = sga(trans_prob,x0)
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
    
    
    alpha = bisection_linesearch(Ef,-prob,trans_prob,grad);
    Ef = Ef + alpha*grad;
    
    iter = iter + 1;
    
    t2=toc;
    fprintf('iteration takes %f seconds\n',t2);
end
end


