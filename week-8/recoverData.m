function X_rec = recoverData(Z, U, K)
  %RECOVERDATA Recovers an approximation of the original data when using the
  %projected data
  %   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the
  %   original data that has been reduced to K dimensions. It returns the
  %   approximate reconstruction in X_rec.

  for i = 1:length(Z)
    X_rec(i, :) = (U(:, 1:K) * Z(i, :)')(:);
  end
end
