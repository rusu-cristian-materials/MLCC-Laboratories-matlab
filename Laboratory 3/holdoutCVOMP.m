function [it, Vm, Vs, Tm, Ts] = holdoutCVOMP(X, Y, perc, nrip, intIter)
%[l, s, Vm, Vs, Tm, Ts] = holdoutCVOMP(algorithm, X, Y, kernel, perc, nrip, intRegPar, intKerPar)
% X: the training examples
% Y: the training labels
% perc: fraction of the dataset to be used for validation
% nrip: number of repetitions of the test for each couple of parameters
% intIter: range of iteration for the Orthogonal Matching Pursuit
%
% Output:
% it: the number of iterations of OMP that minimize the classification
% error on the validation set
% Vm, Vs: median and variance of the validation error for each couple of parameters
% Tm, Ts: median and variance of the error computed on the training set for each couple
%       of parameters
%
% intIter = 1:50;
% [Xtr, Ytr] = MixGauss([[0;0],[1;1]],[0.5,0.25],100);
% Xtr = [Xtr 0.01*randn(200, 28)];
% [l, s, Vm, Vs, Tm, Ts] = holdoutCVOMP(Xtr, Ytr, 0.5, 5, intIter);

    nIter = numel(intIter);
 
    
    n = size(X,1);
    ntr = ceil(n*(1-perc));
    
    tmn = zeros(nIter, nrip);
    vmn = zeros(nIter, nrip);
    
    for rip = 1:nrip
        I = randperm(n);
        Xtr = X(I(1:ntr),:);
        Ytr = Y(I(1:ntr),:);
        Xvl = X(I(ntr+1:end),:);
        Yvl = Y(I(ntr+1:end),:);
        
        iit = 0;    
        for it=intIter
            iit = iit + 1;
            [w, ~, ~] = OMatchingPursuit(Xtr, Ytr, it);
            tmn(iit, rip) =  calcErr(Xtr*w,Ytr);
            vmn(iit, rip)  = calcErr(Xvl*w,Yvl);

            %str = sprintf('rip\tIter\tvalErr\ttrErr\n%d\t%d\t%f\t%f\n',rip, it, vmn(iit,rip), tmn(iit,rip));
            %disp(str);
        end
       
    end
    
    Tm = median(tmn,2);
    Ts = std(tmn,0,2);
    Vm = median(vmn,2);
    Vs = std(vmn,0,2);
    
    row = find(Vm <= min(min(Vm)));
    
    it = intIter(row(1));
end

function err = calcErr(T, Y)
    err = mean(sign(T)~=sign(Y));
end