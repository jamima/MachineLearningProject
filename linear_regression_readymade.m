%% Linear regression using readymade Matlab function "regress"
%
% Start from scratch
clear all;

% Load the training data
data = csvread('regression_dataset_training.csv',1);
train_param = data(:,2:51);
train_class = data(:,52);
[N,p] = size(train_param);

% Fit linear models to each individual feature (word)
mdl_reg = zeros(p,2);
for i = 1:p
    mdl_reg(i,:) = regress(train_class,[ones(N,1) train_param(:,i)]);
end

%% Generate a prediction with the fitted models and calculate fit error
% on training data
pred_train = zeros(N,p);
train_mse = zeros(1,p);
for i = 1:p
    pred_train(:,i) = mdl_reg(i,1) + train_param(:,i) * mdl_reg(i,2);
    train_mse(i) = sum((pred_train(:,i) - train_class).^2) / size(train_class,1);
end

% Visualize the mean square error of the predictions
figure
plot(train_mse,'o')
xlabel('Features')
ylabel('MSE value')
title('MSE of train data')

%% Time to load test data and test our model
test_param = csvread('regression_dataset_testing.csv',1,1);
test_class = csvread('regression_dataset_testing_solution.csv',1,1);

% Generate a prediction with the fitted models and calculate fit error
pred = zeros(size(test_param,1),p);
test_mse = zeros(1,p);
for i = 1:p
    pred(:,i) = mdl_reg(i,1) + test_param(:,i) * mdl_reg(i,2);
    test_mse(i) = sum((pred(:,i) - test_class).^2) / size(test_class,1);
end

%% Feature intercepts and weights
figure
bar(1:size(data,2)-2, mdl_reg(:,1))
xlabel('w_0');
ylabel('weight value');
title('Gradient descent intercept');
figure

bar(1:size(data,2)-2, mdl_reg(:,2))
xlabel('w_1');
ylabel('weight value');
title('Gradient descent weights');

%% Visualize the mean square error of the predictions
figure
plot(test_mse,'o')
xlabel('Features')
ylabel('MSE value')
title('MSE of test data')