%% Start clean
clear all

%% First load training data
train_data = csvread('classification_dataset_training.csv',1);

% Stripe down the parameters and class of the training data
train_param = train_data(:,2:51);
train_class = train_data(:,52);
[N,f] = size(train_param);

% Calculate w's
alpha = 1;
K = 2;
p_r = sum(train_class) / N;
p_0j = zeros(1,f);
p_1j = zeros(1,f);
for i=1:f
    p_0j(i) = (alpha + sum(train_param(~logical(train_class),i))) / (K * alpha +...
        sum(~logical(train_class)));
    p_1j(i) = (alpha + sum(train_param(logical(train_class),i))) / (K * alpha +...
        sum(logical(train_class)));
end

% Calculate weights w_j
w_j = log(p_0j .* (1 - p_1j) ./ (p_1j .* (1 - p_0j)));
w_0 = log((1 - p_r) / p_r) + sum(log((1 - p_0j) ./ (1 - p_1j)));
w = [w_0 w_j];

% Predict classes of training data and calculate classification errors
train_pred = 1 ./ (1 + exp(w_0 + train_param * w_j')) > 0.5;
train_error = (sum(train_pred ~= train_class) / N) * 100


figure
bar(1:size(train_data,2)-2, w_j)
xlabel('w');
ylabel('weight value');
title('Naive Bayes weights');

%% Test the (trained) classifier on the test data
test_data = csvread('classification_dataset_testing.csv',1);
test_data_sol = csvread('classification_dataset_testing_solution.csv',1);

stars_test = 1 ./ (1 + exp(w_0 + test_data(:,2:51) * w_j')) > 0.5;
% Get errors
test_error = (sum(stars_test ~= test_data_sol(:,2)) / size(test_data,1))*100
test_accuracy = 100 - test_error