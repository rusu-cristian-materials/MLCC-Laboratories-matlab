function K = Gram(X1, X2, func, par)
    if strcmp(func,'l2dist')
        K = SquareDist(X1, X2);
    elseif strcmp(func,'gaussian')
        K = exp(-SquareDist(X1, X2)/(2*par^2));
    elseif strcmp(func,'polynomial')
        K = (1 + X1*X2').^par;
    else
        msgbox(['The function: "' func '" is not present in the method Gram']);
    end
end
