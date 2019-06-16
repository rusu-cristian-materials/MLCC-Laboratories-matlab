function Y = flipLabels(Y, p)
% function [Yn] = flipLabels(Y, p)
% flips p percent of labels to be flipped
% p is in [0,100]
% the labels must be +1 and -1
    
    msg = 'Ytr should be an array of +1 and -1';
    if ~all(abs(Y(:)) == 1)
        error(msg) ;
    end

    if p < 1
        disp('Warning: percent is in the range [0, 100], not [0, 1]') ;
    end
    
    n=numel(Y);
    n_flips = floor(n * p / 100);

    I = randperm(n);

    sel = I(1:n_flips);
    % Y(sel) = 1 -1 * Y(sel); This would be the function for 0 and 1 labels
    Y(sel) = - Y(sel); % For +1 and -1 labels
end