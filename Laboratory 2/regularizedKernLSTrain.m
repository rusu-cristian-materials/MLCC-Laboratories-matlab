function c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, lambda)
%
% function c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, lambda)
%
% INPUT PARAMETERS
%   Xtr training input
%   Ytr training output
%   kernel type of kernel: 'linear', 'polynomial', 'gaussian'
%   lambda regularization parameter
%
% OUTPUT PARAMETERS
%   c model weights
%
% EXAMPLE
%   c =  regularizedKernLSTrain(Xtr, Ytr, 'gaussian', 1, 1e-1);

    n = size(Xtr,1);
    K = KernelMatrix(Xtr, Xtr, kernel, sigma);
    c = (K + lambda * n * eye(n)) \ Ytr;
end
