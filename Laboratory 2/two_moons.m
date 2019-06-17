function [Xtr, Ytrn, Xts, Ytsn] = two_moons(npoints, pflip)
    load('moons_dataset.mat');
    npoints = min(100,npoints);
    I = randperm(100);
    sel = I(1:npoints);
    Xtr = Xtr(sel,:);
    Ytrn = flipLabels(Ytr(sel), pflip);
    Ytsn = flipLabels(Yts, pflip);
end
