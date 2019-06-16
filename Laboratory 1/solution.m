close all
clear
clc

rng(42);

%%%%%%%%%%%%%%%%%%%%%%%%% Section 1 %%%%%%%%%%%%%%%%%%%%%%%%%

%% 1.A
help MixGauss

%% 1.B
[X, Y] = MixGauss([[0;0], [1;1]],[0.5, 0.25], 50);
Y = 2*mod(Y, 2)-1;
figure;
subplot(2, 2, 1); scatter(X(:,1), X(:,2), 50, Y, 'filled'); % type "help scatter" to see what the parameters mean
title('dataset 1');

%% 1.C
Ntr = 500; % number of data points
% a four class problem
c1 = [0; 0]; c2 = [1; 0]; c3 = [0; 1]; c4 = [1; 1]; % declare all the centers
sigma = 0.3; % standard deviation
[Xtr, Ytr] = MixGauss([c1 c3 c4 c2], sigma*ones(1,4), Ntr);
subplot(2, 2, 2); scatter(Xtr(:,1), Xtr(:,2), 50, Ytr, 'filled'); title('dataset 2');

% let us turn the problem to a two class problem, the xor problem
Ytr = 2*mod(Ytr, 2)-1;
subplot(2, 2, 3); scatter(Xtr(:,1), Xtr(:,2), 50, Ytr, 'filled'); title('dataset 3 (train dataset)');

%% 1.D
Nts = 50; % this should be a fraction of Ntr
[Xts, Yts] = MixGauss([c1 c3 c4 c2], sigma*ones(1,4), Nts); % same centers and sigmas
Yts = 2*mod(Yts, 2)-1;
subplot(2, 2, 4); scatter(Xts(:,1), Xts(:,2), 50, Yts, 'filled'); title('dataset 4 (test dataset)');

%%%%%%%%%%%%%%%%%%%%%%%%% Section 2 %%%%%%%%%%%%%%%%%%%%%%%%%

% starts with a fresh figure
figure;

%% 2.A
help kNNClassify

%% 2.B
K = 9; % we are doing K-NN classification

%% 2.C1-2.C3
YpredC1 = kNNClassify(Xtr, Ytr, K, Xts);
subplot(2, 1, 1); scatter(Xts(:,1), Xts(:,2), 50, Yts, 'filled'); %plot test points (filled circles) associating a different color to each "true" label
hold on;
subplot(2, 1, 1); scatter(Xts(:,1), Xts(:,2), 70, YpredC1, 'o'); % plot test points (empty circles) associating a different color to each estimated label
title(['KNN prediction with K = ' num2str(K)]);

errorC1 = sum(YpredC1 ~= Yts) ./ size(Yts, 1);
% be careful, YpredC1 can have "0" if K is even

help separatingFkNN

subplot(2, 1, 2); separatingFkNN(Xtr, Ytr, K);
hold on
subplot(2, 1, 2); scatter(Xts(:,1), Xts(:,2), 25, Yts);
title(['countours of classification, K = ' num2str(K)]);

%%%%%%%%%%%%%%%%%%%%%%%%% Section 3 %%%%%%%%%%%%%%%%%%%%%%%%%

% starts with a fresh figure
figure;

help holdoutCVkNN

%% 3.A
N = size(Xtr, 1);
perc = 10; % percentage to use for hold-out
nrep = 5; % number of repetitions
K_interval = 2:10;

[K_best, Vm, Vs, Tm, Ts] = holdoutCVkNN(Xtr, Ytr, perc, nrep, K_interval);

subplot(2, 2, 1); errorbar(K_interval, Tm, sqrt(Ts), '-bo', 'LineWidth', 2);
hold on; subplot(2, 2, 1); errorbar(K_interval, Vm, sqrt(Vs), '-rp', 'LineWidth', 2);
legend('Test', 'Validation');
xlabel('k');
ylabel('error');
title('3.A hold out, initial dataset');

%% 3.B
p = 10; % percentage of error, how many labels to flip
Ytr_noisy = flipLabels(Ytr, p); % randomly flip some of the correct labels

[K_best_noisy, Vm, Vs, Tm, Ts] = holdoutCVkNN(Xtr, Ytr_noisy, perc, nrep, K_interval);

subplot(2, 2, 2); errorbar(K_interval, Tm, sqrt(Ts), '-bo', 'LineWidth', 2);
hold on; subplot(2, 2, 2); errorbar(K_interval, Vm, sqrt(Vs), '-rp', 'LineWidth', 2);
legend('Test', 'Validation');
xlabel('k');
ylabel('error');
title(['3.B hold out, noisy dataset, p = ' num2str(p) ', perc = ' num2str(perc)]);

%% 3.C
K = 5;
p_interval = 0:5:25; % because we start from 0 here, there will be a warning
perc = 10;
nrep = 3;
Vms = zeros(length(p_interval), 1); Tms = zeros(length(p_interval), 1);
Vss = zeros(length(p_interval), 1); Tss = zeros(length(p_interval), 1);

index = 0;
for p = p_interval
    index = index + 1;
    Ytr_noisy = flipLabels(Ytr, p); % randomly flip some of the correct labels
    
    [~, Vm, Vs, Tm, Ts] = holdoutCVkNN(Xtr, Ytr_noisy, perc, nrep, K);
    Vms(index) = Vm; Tms(index) = Tm;
    Vss(index) = Vs; Tss(index) = Ts;
end

subplot(2, 2, 3); errorbar(p_interval, Tms, sqrt(Tss), '-bo', 'LineWidth', 2);
hold on; subplot(2, 2, 3); errorbar(p_interval, Vms, sqrt(Vss), '-rp', 'LineWidth', 2);
legend('Test', 'Validation');
xlabel('p');
ylabel('error');
title(['3.C hold out, noisy dataset, K = ' num2str(K) ', nrep = ' num2str(nrep) ', perc = ' num2str(perc)]);

%%% do the same for perc and nrep

%% 3.D
K_interval = 1:20;
p = 5; % percentage of error, how many labels to flip
Yts_noisy = flipLabels(Yts, p); % Yte was generated at 1.D

errors = zeros(length(K_interval), 1);
index = 0;
for K = K_interval
    index = index + 1;
    Ypred = kNNClassify(Xtr, Ytr, K, Xts);
    errors(index) = sum(Ypred ~= Yts_noisy) ./ size(Yts_noisy, 1);
end

subplot(2, 2, 4); plot(K_interval, errors, '-bo', 'LineWidth', 2);
xlabel('k');
ylabel('error');
title(['3.D test error, p = ' num2str(p)]);
