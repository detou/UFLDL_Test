function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% forward prop
%%% YOUR CODE HERE %%%

for i = 1 : numHidden + 1
    if i == 1
        hAct{i} = bsxfun(@plus, stack{i}.W * data, stack{i}.b);
    else
        hAct{i} = bsxfun(@plus, stack{i}.W * hAct{i - 1}, stack{i}.b);
    end
    
    % Apply activation function to hAct
    switch ei.activation_fun
        case 'logistic'
            hAct{i} = sigmoid(hAct{i}); 
    end
end

% Normalize probabilities
pred_prob = bsxfun(@rdivide, exp(hAct{numHidden + 1}), sum(exp(hAct{numHidden + 1}), 1));

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

% Get a matrix with ones at i, k where y(i) == k
I = sub2ind(size(pred_prob), labels', 1:size(pred_prob,2));
O = zeros(size(pred_prob));
O(I) = 1;

% Calculate cost (WARNING: not sure about it but seems to look good...)
ceCost = 1 / (2 * size(data, 2)) * sum(sqrt(sum((O - pred_prob) .^ 2)));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

delta = cell(numHidden + 1, 1);

for i = numHidden + 1 : -1 : 1
   if i == numHidden + 1
      delta{i} = stack{i}.W' * pred_prob; 
   else
      delta{i} = stack{i}.W' * delta{i + 1} * hAct{i} * (1 - hAct{i});
   end
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



