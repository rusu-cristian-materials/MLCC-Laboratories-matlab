function separatingFKernRLS(c, Xtr, Ytr, kernel, sigma, Xts)
% function separatingFKernRLS(c, Xtr, kernel, sigma, Xts)
% the function classifies points evenly sampled in a visualization area,
% according to the classifier Regularized Least Squares
%
% c - coefficents of the function
% Xtr - training examples
% Ytr - training labels
% kernel, sigma - parameters used in learning the function
% Xts - test examples on which to plot the separating function
%
% lambda = 0.01;
% kernel = 'gaussian';
% sigma = 1;
% [Xtr, Ytr] = MixGauss([[0;0],[1;1]], [0.5,0.25], 1000);
% [Xts, Yts] = MixGauss([[0;0],[1;1]], [0.5,0.25], 1000);
% Ytr(Ytr==2) = -1;
% Yts(Yts==2) = -1;
% c = regularizedKernLSTrain(Xtr, Ytr, kernel, sigma, lambda);
% separatingFKernRLS(c, Xtr, Ytr, kernel, sigma, Xts);

    step = 0.05;
    x = min(Xts(:,1)):step:max(Xts(:,1));
    y = min(Xts(:,2)):step:max(Xts(:,2));
    [X, Y] = meshgrid(x, y);
    XGrid = [X(:), Y(:)];

    YGrid = regularizedKernLSTest(c, Xtr, kernel, sigma, XGrid);
    figure;
    scatter(Xtr(:,1), Xtr(:,2), 40, Ytr, 'filled');
    hold on
    contour(x, y, reshape(YGrid,numel(y),numel(x)),[0;0]);
    hold off
end
