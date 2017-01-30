% Clean start
clear all
clc


data = csvread('regression_dataset_training.csv',1);
train_param = data(:,2:51);
train_class = data(:,52);

% From Modeling and Estimation course: (almost) always normalize data
train_mean = mean(train_param);
train_std = std(train_param);

train_param = bsxfun(@minus, train_param, train_mean);
train_param = bsxfun(@rdivide, train_param, train_std);
train_param = [ones(size(train_param,1),1) train_param];

% The remaining data is used for testing and is also normalized
test_data = csvread('regression_dataset_testing.csv',1);

test_param = test_data(:,2:51);
test_param = bsxfun(@minus,test_param,train_mean);
test_param = bsxfun(@rdivide,test_param,train_std);
test_param = [ones(size(test_param,1),1) test_param];

test_class = csvread('classification_dataset_testing_solution.csv',1,1);

%% Now that the working data is prepared, run gradient descent implemented
% as a function with given initial values to w. In this case, zeros
w = zeros(51,1);
n_step = 0.01;
w = [ones(51,1) w];
[w_out,flag] = gradient_descent(train_param,train_class,w,n_step);

%% Feature weights
figure
bar(0:size(data,2)-2, w_out)
xlabel('w_j');
ylabel('weight value');
title('logistic discriminant weights');

%% Use only largest values
N_large = 5;
w_temp = w_out;
w_cs = zeros(51,1);
for n = 1:N_large
    [~,I] = max(abs(w_temp));
    w_cs(I,1) = w_temp(I,1);
    w_temp(I,1) = 0;
end

%% Calculate classification error for both training and test data
estim_train_cs = train_param*w_out;
estim_test_cs = test_param*w_out;
estim_test_prob_cs = test_param*w_cs;
histogram(estim_train_cs)




