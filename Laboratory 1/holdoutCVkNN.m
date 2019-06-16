function [k, Vm, Vs, Tm, Ts] = holdoutCVkNN(X, Y, perc, nrep, intK)
% X: the dataset (test set excluded)
% Y: the labels (test set excluded)
% perc: percentage of the dataset to be used for validation (in [0,100])
% nrep: number of repetitions of the test for each couple of parameters
% intK: list of regularization parameters
%       for example intK = [1 3 5 7 9 11 17 21 31 41 51 71]
%
% Output:
% k: the value k in intK that minimize the mean of the
%       validation error
% Vm, Vs: mean and variance of the validation error for each couple of parameters
% Tm, Ts: mean and variance of the error computed on the training set for each couple
%       of parameters
%
% Example of code:
% intK = [1 3 5 7 9 11 17 21 31 41 51 71];
% [X, Y] = MixGauss([[0;0],[1;1]], [0.5,0.25], 1000);
% Y(Y==2)=-1;
% [k, Vm, Vs, Tm, Ts] = holdoutCVkNN(X, Y, 50, 10, intK);
% errorbar(intK, Vm, sqrt(Vs), 'b');
% hold on
% errorbar(intK, Tm, sqrt(Ts), 'r');
    if perc < 1
        disp('Warning: percent is in the range [0, 100], not [0, 1]') ;
    end

    nK = numel(intK);

    n = size(X,1);
    ntr = ceil(n * (1 - perc / 100));
    if n * perc / 100 < 1
        error('0 points in the validation set, cannot validate.')
    end
    Tm = zeros(1, nK);
    Ts = zeros(1, nK);
    Vm = zeros(1, nK);
    Vs = zeros(1, nK);

    ym = (max(Y) + min(Y))/2;

    ik = 0;
    for k=intK
        ik = ik + 1;
        for rip = 1:nrep
            I = randperm(n);
            Xtr = X(I(1:ntr),:);
            Ytr = Y(I(1:ntr),:);
            Xvl = X(I(ntr+1:end),:);
            Yvl = Y(I(ntr+1:end),:);

            trError =  calcErr(kNNClassify(Xtr, Ytr, k, Xtr),Ytr, ym);
            Tm(1, ik) = Tm(1, ik) + trError;
            Ts(1, ik) = Ts(1, ik) + trError^2;

            valError  = calcErr(kNNClassify(Xtr, Ytr, k, Xvl),Yvl, ym);
            Vm(1, ik) = Vm(1, ik) + valError;
            Vs(1, ik) = Vs(1, ik) + valError^2;

            %str = sprintf('k\tvalErr\ttrErr\n%f\t%f\t%f\t%f\n', k, valError, trError);
            %disp(str);
        end
    end

    Tm = Tm/nrep;
    Ts = Ts/nrep - Tm.^2;

    Vm = Vm/nrep;
    Vs = Vs/nrep - Vm.^2;

%     idx = find(Vm <= min(Vm(:)));
%     k = intK(idx(1));
    [~, argmin] = min(Vm);
    k = intK(argmin);
end

function err = calcErr(T, Y, m)
    vT = (T >= m);
    vY = (Y >= m);
    err = sum(vT ~= vY)/numel(Y);
end
