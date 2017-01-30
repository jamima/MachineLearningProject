%% Decision Tree
clear all
clc

dt = csvread('classification_dataset_training.csv',1);
train_param = dt(:,2:51);
train_class = dt(:,52);
ctree = fitctree(train_param,train_class)
%view(ctree,'Mode','Graph');
ctree1 = prune(ctree,'Level',0); % 50
[estim_train,score1] = predict(ctree1,train_param);
estim_error_train = sum(estim_train~=train_class) / size(train_class,1) * 100

accuracy_train = 100 - estim_error_train
%view(ctree1,'Mode','Graph');
%ctree = fitctree(train_param,train_class)
%cvctree = crossval(ctree);
%cvloss = kfoldLoss(cvctree)

test_dt = csvread('classification_dataset_testing.csv',1);
test_param = test_dt(:,2:51);
%test_class = test_dt(:,52);
test_class = csvread('classification_dataset_testing_solution.csv',1,1);

[estim_test,score2] = predict(ctree1,test_param);
%estim_test = kfoldpredict(

estim_error_test = sum(estim_test~=test_class) / size(test_class,1) * 100

accuracy_test = 100 - estim_error_test