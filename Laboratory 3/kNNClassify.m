function Ypred = kNNClassify(Xtr, Ytr, k, Xte)
%
% function Ypred = kNNClassify(Xtr, Ytr, k, Xte)
%
% INPUT PARAMETERS
%   Xtr training input
%   Ytr training output 
%   k number of neighbours
%   Xte test input
% 
% OUTPUT PARAMETERS
%   Ypred estimated test output
%
% EXAMPLE
%   Ypred = kNNClassify(Xtr, Ytr, 5, Xte);

    n = size(Xtr,1);
    m = size(Xte,1);
    
    msg = 'Ytr should be an array of +1 and -1';
    if ~all(abs(Ytr(:)) == 1)
        error(msg) ;
    end
    
    if k > n
        disp('k is greater than number of points n, setting k = n') ;
        k = n;
    end
    
    Ypred = zeros(m,1);

    DistMat = SquareDist(Xtr, Xte);
    
    for j= 1:m
        SortdDistMat = DistMat(:,j);
        
        [~, I] = sort(SortdDistMat);
        idx = I(1:k);
        val = sum(Ytr(idx))/k;

        Ypred(j) = sign(val);
    end
end
