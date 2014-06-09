function [J, grad] = cofiCostFunc(params, ...
                                  Y, ...
                                  R, ...
                                  num_users, ...
                                  num_movies, ...
                                  num_features, ...
                                  lambda)
  %COFICOSTFUNC Collaborative filtering cost function
  %   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
  %   num_features, lambda) returns the cost and gradient for the
  %   collaborative filtering problem.

  % Unfold the U and W matrices from params
  X     = reshape(params(1:(num_movies * num_features)), num_movies, num_features);
  Theta = reshape(params((num_movies * num_features + 1):end), num_users, num_features);

  VarianceCost            = (1 / 2) * sum(((((X * Theta') - Y) .^ 2) .* R)(:));
  ThetaRegularizationCost = (lambda / 2) * sum((Theta .^ 2)(:));
  XRegularizationCost     = (lambda / 2) * sum((X .^ 2)(:));

  J          = VarianceCost + ThetaRegularizationCost + XRegularizationCost;
  X_grad     = (((X * Theta') - Y) .* R) * Theta + (lambda * X);
  Theta_grad = (((X * Theta') - Y) .* R)' * X + (lambda * Theta);
  grad       = [X_grad(:); Theta_grad(:)];
end
