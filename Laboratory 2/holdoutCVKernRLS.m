function [l, s, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(X, Y, kernel, perc, nrip, intLambda, intKerPar)
%[l, s, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(X, Y, kernel, perc, nrip, intLambda, intKerPar)
% Xtr: the training examples
% Ytr: the training labels
% kernel: the kernel function (see help Gram).
% perc: percentage of the dataset to be used for validation
% nrip: number of repetitions of the test for each couple of parameters
% intLambda: list of regularization parameters
%       for example intLambda = [5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001];
% intKerPar: list of kernel parameters
%       for example intKerPar = [10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01];
%
% Output:
% l, s: the couple of lambda and kernel parameter that minimize the median of the
%       validation error
% Vm, Vs: median and variance of the validation error for each couple of parameters
% Tm, Ts: median and variance of the error computed on the training set for each couple
%       of parameters
%
% Example of code:
% intLambda = [5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001];
% intKerPar = [10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01];
% [Xtr, Ytr] = MixGauss([[0;0],[1;1]],[0.5,0.25],100);
% [l, s, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(Xtr, Ytr,'gaussian', 0.5, 5, intLambda, intKerPar);
% plot(intLambda, Vm, 'b');
% hold on
% plot(intLambda, Tm, 'r');
    nKerPar = numel(intKerPar);
    nLambda = numel(intLambda);


    n = size(X,1);
    ntr = ceil(n*(1-perc));

    Tm = zeros(nLambda, nKerPar);
    Ts = zeros(nLambda, nKerPar);
    Vm = zeros(nLambda, nKerPar);
    Vs = zeros(nLambda, nKerPar);

    ym = (max(Y) + min(Y))/2;

    il = 0;
    for l=intLambda
        il = il + 1;
        is = 0;
        for s=intKerPar
            is = is + 1;
            trerr = zeros(nrip,1);
            vlerr = zeros(nrip,1);
            for rip = 1:nrip
                I = randperm(n);
                Xtr = X(I(1:ntr),:);
                Ytr = Y(I(1:ntr),:);
                Xvl = X(I(ntr+1:end),:);
                Yvl = Y(I(ntr+1:end),:);

                w = regularizedKernLSTrain(Xtr, Ytr, kernel, s, l);

                trerr(rip) =  calcErr(regularizedKernLSTest(w, Xtr, kernel, s, Xtr), Ytr, ym);
                vlerr(rip)  = calcErr(regularizedKernLSTest(w, Xtr, kernel, s, Xvl), Yvl, ym);

%                 str = sprintf('l\ts\tvalErr\ttrErr\n%f\t%f\t%f\t%f\n', l, s, vlerr(rip), trerr(rip));
%                 disp(str);

                %fprintf('l: %.2d, s: %.3f, valErr: %.3f, trErr: %.3f\n',...
                %l, s, vlerr(rip), trerr(rip));
            end
            Tm(il, is) = median(trerr);
            Ts(il, is) = std(trerr);
            Vm(il, is) = median(vlerr);
            Vs(il, is) = std(vlerr);
        end
    end

    %[row, col] = find(Vm + sqrt(Vs) <= min(min(Vm+sqrt(Vs))));
    [row, col] = find(Vm <= min(min(Vm)));

    l = intLambda(row(1));
    s = intKerPar(col(1));
end
function err = calcErr(T, Y, m)
    vT = (T >= m);
    vY = (Y >= m);
    err = sum(vT ~= vY)/numel(Y);
end
