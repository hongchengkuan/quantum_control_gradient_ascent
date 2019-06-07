function V = V_Morse(x,D,a,x0)
    V = D*(exp(-a*(x-x0))-1).^2 -D;
end
