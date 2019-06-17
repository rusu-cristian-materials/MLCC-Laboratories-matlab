function y = regularizedKernLSTest(c, Xtr, kernel, sigma, Xts)
%
% function y = regularizedKernLSTest(c, Xtr, kernel, sigma, Xts)
%
% INPUT PARAMETERS
%   c model weights
%   Xtr training input
%   kernel type of kernel: 'linear', 'polynomial', 'gaussian'
%   sigma width of the gaussian kernel, if used
%   Xts test points
%
%
% OUTPUT PARAMETERS
%   y predicted model values
%
% EXAMPLE
%   y =  regularizedKernLSTest(c, Xtr, 'gaussian', 1, Xts);

    Ktest = KernelMatrix(Xts, Xtr, kernel, sigma);
    y = Ktest * c;
end
