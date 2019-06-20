function [V, d, X_proj] = PCA(X, k)
% [V, d, X_proj] = PCA(X, k)
% computes the first k eigenvectors, eigenvalues and projections of the 
% matrix X'*X/n where n is the number of rows in X.
%
% X is the dataset
% k is the number of components
%
% V is a matrix of the form [v_1, ..., v_k] where v_i is the i-th
% eigenvector
% d is the list of the first k eigenvalues
% X_proj is the projection of X on the linear space spanned by the
% eigenvectors in V
    n = size(X,1);
    [V, D] = eigs(X' * X/n, k);
    d = diag(D);
    d = d.*(d>0);
    [d,I] = sort(d , 'descend');
    V = V(:,I);
    X_proj = X*V;
end


