function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% m = number of examples
% n = number of features

n = size(X, 2);

z = X * theta;
p = predict(theta, X);
g = sigmoid(z);

% For a binary classification
J = (1 / m) * sum(-y .* log(g) - (1 - y) .* log(1 - g)) + lambda / (2 * m) * (sum(theta .^ 2) - theta(1) .^ 2);

grad = (1 / m) * X' * (g - y) + lambda / m * theta;
grad(1) = (1 / m) * X(: , 1)' * (g - y);


% grad(1) = (1 / m(1)) * X' * (g(1) - y(1))




% =============================================================

end
