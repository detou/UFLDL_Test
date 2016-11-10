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
    if i < numHidden + 1 % => Do not apply to the last layer
        switch ei.activation_fun
            case 'logistic'
                hAct{i} = sigmoid(hAct{i}); 
        end
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
ceCost=-sum(log(pred_prob(I)));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

% Compute delta
delta = cell(numHidden + 1, 1);

O = zeros(size(pred_prob));
O(I) = 1;

for i = numHidden + 1 : -1 : 1
   fd = hAct{i} .* (1 - hAct{i});
   if i == numHidden + 1
      delta{i} = - (O - pred_prob); %.* fd; => Does not work with this term... Error in tutorial ?
   else
      delta{i} = stack{i + 1}.W' * delta{i + 1} .* fd;
   end
end

% Compute gradient
for i = 1 : numHidden + 1
    if(i == 1)
        gradStack{i}.W = delta{i} * data';
    else
        gradStack{i}.W = delta{i} * hAct{i - 1}';
    end
    gradStack{i}.b = sum(delta{i}, 2);
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for i = 1: numHidden + 1
    wCost = wCost + 0.5 * ei.lambda * sum (stack{i}.W(:) .^ 2);% The sum of squares of the weights
end

cost = ceCost + wCost;

% Computing the gradient of the weight decay.
for i = numHidden: -1: 1
    gradStack{i}.W = gradStack{i}.W + ei.lambda * stack{i}.W;% softmax Weak weight attenuation
end


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



