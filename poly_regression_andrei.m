% Start from scratch
clear all;

% Load the training data
data = csvread('regression_dataset_training.csv',1);
train_param = data(:,2:51);
train_class = data(:,52);

% Define the highest polynomial degree to try
poly_degree = 5;

% Info about the data length
N = length(train_param);

% Create basis functions with biased column of 1s and up to polynomials
% of degree poly_degree
b_funs = bsxfun(@power,train_param(:,3),0:poly_degree);

% Train the model with data from training data, spliting it with "k folds"
% and also save the predicted values of validation.
% Selecting samples for training / validation
k = 100;
p = randperm(100);

% Prealocate variables for phi and validation predictions.
omega = cell((poly_degree+1),1);
validation_predicts = zeros(N,(poly_degree+1));

for i=1:100
    % Selecting LOO folds:
    training_idx = p([1:i-1 i+1:end]);
    test_idx = p(i);
    
    for j=1:(poly_degree+1)
        phi_temp = b_funs(training_idx, 1:j);
        omega_temp = pinv(phi_temp) * train_param(training_idx,2);
        omega{j} = omega_temp;
        validation_predicts(test_idx,j) = b_funs(test_idx, 1:j) * omega_temp;
    end
end

% Validation errors
validation_mse = (sum((validation_predicts - repmat(train_class,[1,poly_degree+1])).^2)) ./ size(train_class,1);

% Chose best degree (lowest validation error)
[~,I] = min(validation_mse)


%% Time to load test data and test our model
test_param = csvread('regression_dataset_testing.csv',1);
test_class = csvread('regression_dataset_testing_solution.csv',1,1);

% Generate basis functions of test data
b_funs_test = bsxfun(@power,test_param(:,1),0:poly_degree);

% Predictions made by our model for test data
N_test = size(test_param,1);
test_predicts = zeros(N_test,(poly_degree+1));
for j=1:(poly_degree+1)
    test_predicts(:,j) = b_funs_test(:,1:j) * omega{j};
end

% Get average squared errors from above predictions
test_errors = sum((test_predicts - repmat(test_class,[1,(poly_degree+1)])).^2) ./ N_test;

%plot(test_data(:, 1), test_data(:, 2), '.')
%hold on
%plot(test_data(:, 1), b_funs_test(:, 1:I) * omega{I}, 'go')