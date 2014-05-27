function [J, grad] = lrCostFunction(theta, X, y, lambda)
  %LRCOSTFUNCTION Compute cost and gradient for logistic regression with
  %regularization
  %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters.

  m = length(y);

  J    = (- (1 / m) * ((log(h(theta, X))' * y) + (log(1 - h(theta, X))' * (1 - y)))) + ...
         ((lambda / (2 * m)) * (theta(2:end)' * theta(2:end)));
  grad = (1 / m) * X' * (h(theta, X) - y) + ((lambda / m) * [[0], theta(2:end)']');
end

function x = h(theta, X)
  x = sigmoid(X * theta);
end
