function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% 1. Randomly Initialize Weights (Thetas)
% 2. Implement Forward Propagation
% 3. Implement code to compute J(Q)
% 4. Implement Backpropagation to compute d/dQ J(Q)


% Steps 1- 3
a_1 = [ones(size(X, 1), 1), X];
z_2 = a_1 * Theta1';
a_2 = [ones(size(z_2, 1), 1), sigmoid(z_2)];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
h_x = sigmoid(z_3);

K = size(Theta2, 1);

% We want y as one-hot code
y_oh = zeros(size(y, 1), K);
a_3_oh = zeros(size(y, 1), K);
[dummy, a_3] = max(a_3, [], 2);


for i = 1 : m;
    for j = 1: K;
        if(j == y(i))
            y_oh(i, j) = 1;
        end

        if(j = a_3(i))
            a_3_oh(i, j) = 1;
        end
    endfor
endfor


J = (1 / m) * sum(sum(-y_oh .* log(h_x) - (1 - y_oh) .* log(1 - h_x)));

% Cost with Regularization

thetaVec = [Theta1(:); Theta2(:)];

J = (1 / m) * sum(sum(-y_oh .* log(h_x) - (1 - y_oh) .* log(1 - h_x))) + ...
lambda / (2 * m) * (sum(thetaVec .^ 2) - sum(Theta1(:, 1) .^ 2) - sum(Theta2(:, 1) .^ 2)); 

% Step 4: Back Propagation
Delta = zeros();
delta_3 = zeros(size(y_oh));
delta_2 = zeros(size(y_oh));

for i = 1 : m;
    % Perform Forward propagation
    % We already have the paramters from above
    delta_3(i, :) = a_3_oh(i, :) - y_oh(i, :);
    delta_2 = delta_3 * Theta2  .* sigmoidGradient(z_2);

endfor

delta_3



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


