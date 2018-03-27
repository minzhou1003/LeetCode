function [Y_predict] = minzhou_RDA_test(X_test, RDAmodel, numofClass)
%
% Testing for RDA
%
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix 
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% 1. Get the training parameters
Mean = RDAmodel.Mu;
covar = RDAmodel.Sigmapooled;

% 2. PLugin the RDA rule and find argmax
% class0:
RDAclass0 = Mean(1,:) /covar * X_test' - 0.5*Mean(1,:)/covar*Mean(1,:)';
% class1:
RDAclass1 = Mean(2,:) /covar * X_test' - 0.5*Mean(2,:)/covar*Mean(2,:)';
[~, argmax] = max([RDAclass0;RDAclass1]);

% 3. Output the predicted labels
Y_predict = (argmax - 1)';

end
