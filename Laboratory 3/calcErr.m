function err = calcErr(T, Y)
    vT = (T >= 0);
    vY = (Y >= 0);
    err = sum(vT ~= vY)/numel(Y);
end