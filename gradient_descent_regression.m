
% Clean start
clear all


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

test_class = csvread('regression_dataset_testing_solution.csv',1,1);

% Now that the working data is prepared, run gradient descent implemented
% as a function with given initial values to w. In this case, zeros

w = zeros(51,1);
n_step = 0.001;
% Initialize variables
alpha = 0.1;
maxIter = 2000;
relDiff = 1e-10;
i = 0;
flag = 0;

while(1)
    % Make prediction using current weights
    train_pred = train_param * w;
    %train_pred = train_param * [w0; w];
    % Calculate gradient as simple rest between real class and prediction
    grad = ((train_pred - train_class)' * train_param)' ./ 5000;
    %grad0 = (train_pred - train_class)'./5000;
    % Update current guess
    w = w - alpha * grad;
    %w0 = w0 - alpha * grad0;
    % Check for convergence
    if abs(grad) < relDiff
        break;
    end
    if i > maxIter
        flag = 1;
        break;
    end
    i = i + 1;
end

% Calculate classification error for both training and test data
train_mse = sum((train_pred - train_class).^2) / size(train_class,1)

test_pred =  test_param * w;
test_mse = sum((test_pred - test_class).^2) / size(test_class,1)

%% Feature weights
figure
bar(0:size(data,2)-2, w)
xlabel('w_j');
ylabel('weight value');
title('Gradient descent weights');

%% Use only largest values of weights
N_large = 7;
w_temp = w;
w_cs = zeros(51,1);
for n = 1:N_large
    [~,I] = max(abs(w_temp));
    w_cs(I,1) = w_temp(I,1);
    w_temp(I,1) = 0;
end

%% Calculate mse error for both training and test data
train_mse_cs = sum((train_param*w_cs - train_class).^2) / size(train_class,1)

test_pred =  test_param * w;
test_mse_cs = sum((test_param*w_cs - test_class).^2) / size(test_class,1)

