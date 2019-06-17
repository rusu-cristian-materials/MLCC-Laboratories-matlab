close all
clear
clc

rng(42);

%%%%%%%%%%%%%%%%%%%%%%%%% Section 1 %%%%%%%%%%%%%%%%%%%%%%%%%

%% 1.A
[Xtr, Ytr] = MixGauss([[0;0],[1;1]], [0.5,0.3], 100);
[Xts, Yts] = MixGauss([[0;0],[1;1]], [0.5,0.3], 100);
Ytr(Ytr==2) = -1; Yts(Yts==2) = -1;

%% 1.B
sigma = 0.1;
lambda = 1;
c = regularizedKernLSTrain(Xtr, Ytr, 'gaussian', sigma, lambda);

%% 1.C
separatingFKernRLS(c, Xtr, Ytr, 'gaussian', sigma, Xts);
title(['no noise, sigma = ' num2str(sigma) ' lambda = ' num2str(lambda)]);

%% 1.D
p = 0.1;

sigma = 0.1;
lambda = 1;
Ytr_noisy = flipLabels(Ytr, p);
c_noisy = regularizedKernLSTrain(Xtr, Ytr_noisy, 'gaussian', sigma, lambda);
separatingFKernRLS(c_noisy, Xtr, Ytr_noisy, 'gaussian', sigma, Xts);
title(['noisy, p = ' num2str(p) ' sigma = ' num2str(sigma) ' lambda = ' num2str(lambda)]);

%% 1.E
npoints = 100; % number of training points
pflipped = 0.1; % procentage flipped

[Xtr, Ytr, Xts, Yts] = two_moons(npoints, pflipped);

figure;
subplot(1,2, 1); scatter(Xtr(:,1), Xtr(:,2), 50, Ytr, 'filled'); title('noisy train dataset');
subplot(1,2, 2); scatter(Xts(:,1), Xts(:,2), 50, Yts, 'filled'); title('noisy test dataset');

%% 1.F
sigma = 0.25;
lambda = 1;
c = regularizedKernLSTrain(Xtr, Ytr, 'gaussian', sigma, lambda);
separatingFKernRLS(c, Xtr, Ytr, 'gaussian', sigma, Xts);
title(['two moons, sigma = ' num2str(sigma) ' lambda = ' num2str(lambda)]);

%%%%%%%%%%%%%%%%%%%%%%%%% Section 2 %%%%%%%%%%%%%%%%%%%%%%%%%

help holdoutCVKernRLS

%% 2.C
intKerPar = 0.5;
intLambda = [10, 7, 5, 2, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001];
nrip = 5;
perc = 0.5;

[l, s, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(Xtr, Ytr, 'gaussian', perc, nrip, intLambda, intKerPar);

figure;
semilogx(intLambda, Vm, '-bo', 'LineWidth', 2);
hold on; semilogx(intLambda, Tm, '-rp', 'LineWidth', 2);
legend('Validation', 'Test');

%% 2.E
intKerPar = [10, 7, 5, 4, 3, 2.5, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01];
intLambda = [10, 7, 5, 2, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001];
nrip = 7;
perc = 0.5;

[lambda_best, sigma_best, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(Xtr, Ytr, 'gaussian', perc, nrip, intLambda, intKerPar);
c = regularizedKernLSTrain(Xtr, Ytr, 'gaussian', sigma_best, lambda_best);
separatingFKernRLS(c, Xtr, Ytr, 'gaussian', sigma_best, Xts);
title(['two moons, gaussian, best sigma = ' num2str(sigma_best) ' best lambda = ' num2str(lambda_best)]);

%%%%%%%%%%%%%%%%%%%%%%%%% Section 3 %%%%%%%%%%%%%%%%%%%%%%%%%

%% 3.C
IntKerPar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
intLambda = [5, 2, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001];
nrip = 7;
perc = 0.5;

[lambda_poly_best, sigma_poly_best, Vm, Vs, Tm, Ts] = holdoutCVKernRLS(Xtr, Ytr, 'polynomial', perc, nrip, intLambda, intKerPar);
c = regularizedKernLSTrain(Xtr, Ytr, 'polynomial', sigma_poly_best, lambda_poly_best);
separatingFKernRLS(c, Xtr, Ytr, 'polynomial', sigma_poly_best, Xts);
title(['two moons, polynomial, best degree = ' num2str(sigma_poly_best) ' best lambda = ' num2str(lambda_poly_best)]);

%% 3.D
K_gaussian = KernelMatrix(Xtr, Xtr, 'gaussian', sigma_best);
K_polynomial = KernelMatrix(Xtr, Xtr, 'polynomial', sigma_poly_best);

[~, D_gaus] = eig(K_gaussian);
[~, D_poly] = eig(K_polynomial);

D_gaus = diag(D_gaus); D_gaus(find(D_gaus<0)) = 0;
D_poly = diag(D_poly); D_poly(find(D_poly<0)) = 0;
figure;
semilogy(D_gaus, '-r', 'LineWidth', 2);
hold on; semilogy(D_poly, '-b', 'LineWidth', 2);
legend('Gaussian eigenvalues', 'Polynomial eigenvalues');
