% Start from scratch
clear all;
clc

% Load the training data
data = csvread('regression_dataset_training.csv',1);
train_param = data(:,2:51);
train_class = data(:,52);
[~,d] = size(train_class);
[n,dx] = size(train_param);
X = zeros(n,dx);
for i = 1:size(train_param,2)
    X(:,i) = zscore(train_param(:,i));
end
Xmat = [ones(n,1) X]; % design matrix
Xcell = cell(1,n);
for i = 1:n
    Xcell{i} = [kron([Xmat(i,:)],eye(d))];
end
[beta,sigma,E,V] = mvregress(Xcell,train_class);
se = sqrt(diag(V));
B = reshape(beta,1,51)';
z = E/chol(sigma);
figure()
plot(z,'.')
title('Standardized Residuals')