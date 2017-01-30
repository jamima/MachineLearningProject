%% Start clean
clear all

% First load training data
train_data = csvread('classification_dataset_training.csv',1);

% Stripe down the features (words) and the classes (stars)
stars = train_data(:,52);
words = train_data(:,2:51);
[N,f] = size(words);

% Calculate w's
K = 2;
folds = 4;
Icv = crossvalind('Kfold',size(words,1),folds);
avrg_error = zeros(10001,1);
ind1 = 1;
for alpha = 100:0.1:1000
    for ind2 = 1:folds
        test_dt = (Icv == ind2);
        train_dt = ~test_dt;
        p_r = sum(stars(train_dt)) / N;
        p_0j = zeros(1,f);
        p_1j = zeros(1,f);
        for ind3 = 1:f
            p_0j(ind3) = (alpha + sum(words(~logical(stars(train_dt)),ind3))) / (K * alpha +...
                sum(~logical(stars(train_dt))));
            p_1j(ind3) = (alpha + sum(words(logical(stars(train_dt)),ind3))) / (K * alpha +...
                sum(logical(stars(train_dt))));
        end

        % Calculate weights w_j
        w_j = log(p_0j .* (1 - p_1j) ./ (p_1j .* (1 - p_0j)));
        w_0 = log((1 - p_r) / p_r) + sum(log((1 - p_0j) ./ (1 - p_1j)));
        w = [w_0 w_j];
        % Estimate classes and calculate classification errors on training data
        diag_train = 1 ./ (1 + exp(w_0 + words(test_dt,:) * w_j')) > 0.5;
        wrong_diag_train = (sum(diag_train ~= stars(test_dt)) / size(stars(test_dt),1));
        avrg_error(ind1,1) = avrg_error(ind1,1) + wrong_diag_train;
    end
    ind1 = ind1 + 1;
end

avrg_error = avrg_error ./ folds;

%% Test the classifier on the test data
test_data = csvread('classification_dataset_testing.csv',1);
test_data_sol = csvread('classification_dataset_testing_solution.csv',1);

stars_test = 1 ./ (1 + exp(w_0 + test_data(:,2:51) * w_j')) > 0.5;
% Get errors
wrong_perc = (sum(stars_test ~= test_data_sol(:,2)) / size(test_data,1))*100
accuracy = 100-wrong_perc