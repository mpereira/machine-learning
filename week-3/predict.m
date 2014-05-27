function Z = predict(theta, X)
  %PREDICT Predict whether the label is 0 or 1 using learned logistic
  %regression parameters theta
  %   p = PREDICT(theta, X) computes the predictions for X using a
  %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

  Z = arrayfun(@discretize, h(theta, X));
end

function Z = h(theta, X)
  Z = sigmoid(X * theta);
end

function x = discretize(value)
  threshold = 0.5;
  if value >= threshold, x = 1; elseif value < threshold, x = 0; end
end
