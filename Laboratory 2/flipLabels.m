function Y = flipLabels(Y, p)
% function [Yn] = flipLabels(Y, p)
% flips p percent of labels to be flipped
% the labels must be +1 and -1
    n=numel(Y);
    n_flips = floor(n*p);

    I = randperm(n);

    sel = I(1:n_flips);
    Y(sel) = -1*Y(sel);
end
