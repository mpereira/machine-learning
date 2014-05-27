function [J, grad] = costFunctionReg(theta, X, y, lambda)
  %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
  %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters.

  m = length(y);

  [J_non_reg, grad] = costFunction(theta, X, y);
  J                 = J_non_reg + ...
                      ((lambda / (2 * m)) * (theta(2:end)' * theta(2:end)));
  grad              = [grad(1); grad(2:end) + ((lambda / m) * theta(2:end))];
end
