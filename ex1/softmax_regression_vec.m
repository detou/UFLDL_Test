function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  theta=[theta,zeros(size(theta,1),1)];
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
  % Get theta sums
  T = exp(theta' * X);
  
  % Get P matrix, giving probability for each k for each X sample
  P = bsxfun(@rdivide, T, sum(T, 1));
  
  % Get a matrix giving the indexes of the elements where y(i) == k in P
  I = sub2ind(size(P), y, 1:size(P,2));
  
  % We get a matrix containing only the value of P where y(i) == k
  M = P(I);
  
  % Get a matrix with ones at i, k where y(i) == k
  O = zeros(size(P));
  O(I) = 1;
  
  % Calculate cost
  f = -sum(log(M));
  
  % Calculate gradient
  g = X * P' - X * O';
  
  g(:,end)=[];
  g=g(:); % make gradient a vector for minFunc

