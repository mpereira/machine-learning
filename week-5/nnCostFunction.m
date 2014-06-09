function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, ...
                                   y, ...
                                   lambda)
  % NNCOSTFUNCTION Implements the neural network cost function for a two layer
  % neural network which performs classification
  %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  %   X, y, lambda) computes the cost and gradient of the neural network. The
  %   parameters for the neural network are "unrolled" into the vector
  %   nn_params and need to be converted back into the weight matrices.
  %
  %   The returned parameter grad should be a "unrolled" vector of the
  %   partial derivatives of the neural network.

  Theta1  = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                    hidden_layer_size, (input_layer_size + 1));

  Theta2  = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));
  m       = size(X, 1);
  Delta1  = 0;
  Delta2  = 0;
  J_total = 0;
  yMap    = eye(num_labels);

  for i = 1:m
    % Forward propagation.
    X_i = X(i, :)';
    a1  = [1; X_i];
    z2  = Theta1 * a1;
    a2  = [1; sigmoid(z2)];
    z3  = Theta2 * a2;
    a3  = sigmoid(z3);
    h_i = a3;
    y_i = yMap(y(i), :)';

    % Backpropagation.
    delta3_i = a3 - y_i;
    delta2_i = ((Theta2' * delta3_i) .* a2 .* (1 - a2))(2:end);
    Delta1   = Delta1 + delta2_i * a1';
    Delta2   = Delta2 + delta3_i * a2';

    J_i     = sum(y_i' * log(h_i) + (1 - y_i') * log(1 - h_i));
    J_total = J_total + J_i;
  end

  r = sum([(Theta1(:, 2:end) .^ 2)(:); (Theta2(:, 2:end) .^ 2)(:)]);
  J = (-(1 / m) * J_total) + ((lambda / (2 * m)) * r);

  D1 = (1 / m) * ([Delta1(:, 1), (Delta1 + (lambda * Theta1))(:, 2:end)]);
  D2 = (1 / m) * ([Delta2(:, 1), (Delta2 + (lambda * Theta2))(:, 2:end)]);

  Theta1_grad = D1;
  Theta2_grad = D2;

  grad = [Theta1_grad(:); Theta2_grad(:)];
end
