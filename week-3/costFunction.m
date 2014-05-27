function [J, grad] = costFunction(theta, X, y)
  %COSTFUNCTION Compute cost and gradient for logistic regression
  %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
  %   parameter for logistic regression and the gradient of the cost
  %   w.r.t. to the parameters.

  m    = length(y);
  J    = - (1 / m) * ...
           ((log(h(theta, X))' * y) + (log(1 - h(theta, X))' * (1 - y)));
  grad = (1 / m) * X' * (h(theta, X) - y);
end

function x = h(theta, X)
  x = sigmoid(X * theta);
end
