function [w, r, I] = OMatchingPursuit(X, Y, T)
% function [w, r, I] = OMatchingPursuit(X, Y, T)
% Orthogonal Maching Pursuit
%
% X input data
% Y output labels
% T number of iterations
%
% w estimated coefficients
% r residuals
% I indices
    
    [N, D] = size(X);

    %%% Initialization of residual, coefficient vector and index set I
    r = Y;
    w = zeros(D, 1);
    I = [];
    
    for i = 1:T-1
        
        I_tmp = 1:D;
        
		%%% Select the column of X which most "explains" the residual
        a_max = -1;
        for j = I_tmp
            a_tmp = ((r' * X(:,j))^2)/(X(:,j)' * X(:,j));
            if a_tmp > a_max
                a_max = a_tmp;
                j_max = j;
            end
        end
        
		%%% Add the index to the set of indexes
        if sum(I == j_max) == 0
            I = [I j_max];
        end
        
		%%% Compute the M matrix
        M_I = zeros(D,D);
        for j = I
            M_I(j,j) = 1;
        end
        
        A = M_I * X' * X * M_I;
        B = M_I * X' * Y;
        
		%%% Update w
        w = pinv(A) * B;
        
		%%% Update the residual
        r = Y - X*w;
        
    end
end
