close all
clear
clc

rng(42);

%%%%%%%%%%%%%%%%%%%%%%%%% Section 1 %%%%%%%%%%%%%%%%%%%%%%%%%

N = 100; D = 30;

%% 1.A
[Xtr, Ytr] = MixGauss([[0;0],[1;1]], [0.7,0.7], 100);
[Xts, Yts] = MixGauss([[0;0],[1;1]], [0.7,0.7], 100);
Ytr(Ytr==2) = -1; Yts(Yts==2) = -1;

%% 1.B
scatter(Xtr(:,1), Xtr(:,2), 50, Ytr, 'filled');
hold on; scatter(Xts(:,1), Xts(:,2), 50, Yts);
title('train and test datasets');

%% 1.C
sigma_noise = 0.01;
Xtr_noise=sigma_noise*randn(2*N,D-2);
Xts_noise=sigma_noise*randn(2*N,D-2);

Xtr =[Xtr, Xtr_noise];
Xts =[Xts, Xts_noise];

%%%%%%%%%%%%%%%%%%%%%%%%% Section 2 %%%%%%%%%%%%%%%%%%%%%%%%%

%% 2.A
k = 10; % get 10 principal components
[V, d, X_proj] = PCA(Xtr, k); % compute the PCA of the training set

%% 2.B
figure; scatter(X_proj(:,1), X_proj(:,2), 50, Ytr, 'filled');

%% 2.C
scatter3(X_proj(:,1), X_proj(:,2), X_proj(:,3), 50, Ytr, 'filled');

%% 2.D
disp('Eigenvalues are: ');
disp(sqrt(d(1:10)))
figure; plot(abs(V(:, 1)), 'or');
title('Eigenvector of highest eigenvalue');

%%%%%%%%%%%%%%%%%%%%%%%%% Section 3 %%%%%%%%%%%%%%%%%%%%%%%%%

%% 3.A
% normalize the training data
m = mean(Xtr);
s = std(Xtr);
for i = 1:2*N
	Xtr(i,:) = Xtr(i,:) - m;
	Xtr(i,:) = Xtr(i,:) ./ s;
end

% normalize the test data
for i = 1:2*N
	Xts(i,:) = Xts(i,:) - m;
	Xts(i,:) = Xts(i,:) ./ s;
end

%% 3.B and 3.C
perc = 0.75;
nrip = 20;
intIter = 2:D;
[it_best, Vm, Vs, Tm, Ts] = holdoutCVOMP(Xtr, Ytr, perc, nrip, intIter);

[w, ~, ~] = OMatchingPursuit(Xtr, Ytr, it_best);
Ypred = sign(Xts*w);
error = calcErr(Yts, Ypred);

figure; scatter(1:D, abs(w));
title('the w vector');

%% 3.D
figure; plot(intIter, Tm, 'r');
hold on; plot(intIter, Vm, 'b');
xlabel('number of iterations for OMP');
ylabel('error');
legend('Test', 'Validation');

%%%%%%%%%%%%%%%%%%%%%%%%% Section 4 %%%%%%%%%%%%%%%%%%%%%%%%%

%% 4.B
% generate a fresh dataset
[Xtr, Ytr] = MixGauss([[0;0],[1;1]], [0.7,0.7], 100);
[Xts, Yts] = MixGauss([[0;0],[1;1]], [0.7,0.7], 100);
Ytr(Ytr==2) = -1; Yts(Yts==2) = -1;

% perform the singular value decomposition, i.e., dimensionality reduction
[U, S, V] = svd(Xtr');
Xtr_proj = U(:, 1)'*Xtr'; % project on the first principal component

% filled points are the original points from the dataset
% empty points are projected on the principal component
figure; scatter(Xtr(:,1), Xtr(:,2), 50, Ytr, 'filled');
Z = zeros(2, 2*N);
% project, point by point
for i = 1:2*N
    Z(:, i) = U(:, 1)*Xtr_proj(i);
end
hold on; scatter(Z(1, :), Z(2, :), 50, Ytr);
title('original and projected datasets');

%%% task: pick a point and its projection and draw a line between them

% apply kNN on the projected dataset
k = 5;
Xts_proj = U(:, 1)'*Xts';
YpredkNN = kNNClassify(Xtr_proj', Ytr, k, Xts_proj'); % this now takes place in a 1-dimensional space, i.e., a line

% classification error
errorkNN = sum(YpredkNN ~= Yts) ./ size(Yts, 1);
