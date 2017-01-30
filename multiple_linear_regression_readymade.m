%% Multiple linear regression using Matlab ready-made function "fitlm"
%
% Start from scratch
clear all;

% Load the training data
data = csvread('regression_dataset_training.csv',1);
train_param = data(:,2:51);
train_class = data(:,52);

%% First try with all the features at once
mdl = fitlm(train_param,train_class);

% Get mse error on training data
train_pred = predict(mdl,train_param);
train_mse = sum((train_pred - train_class).^2) / size(train_class,1)

% Time to load test data and test our model
test_param = csvread('regression_dataset_testing.csv',1,1);
test_class = csvread('regression_dataset_testing_solution.csv',1,1);

test_pred = predict(mdl,test_param);
test_mse = sum((test_pred - test_class).^2) / size(test_class,1)

%% Second, trying to use the previously 5 best features identified in
% single linear regression.
best_param = [7];
train_param_best = train_param(:,best_param);
mdl_best = fitlm(train_param_best,train_class);

% Predict and get mse on train data
train_param_best = train_param(:,best_param);
train_pred_best = predict(mdl_best,train_param_best);
train_mse_best = sum((train_pred_best - train_class).^2) / size(train_class,1)

% Predict on the test data
test_param_best = test_param(:,best_param);
test_pred_best = predict(mdl_best,test_param_best);
test_mse_best = sum((test_pred_best - test_class).^2) / size(test_class,1)