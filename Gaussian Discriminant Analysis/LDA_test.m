function [Y_predict] = minzhou_LDA_test(X_test, LDAmodel, numofClass)
%
% Testing for LDA
%
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% LDAmodel : the parameters of LDA classifier which has the follwoing fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test
%
% 1. Get the training parameters
n = size(X_test, 1);
mean = LDAmodel.Mu;
cov = LDAmodel.Sigma;
p = LDAmodel.Pi;

Y_predict = zeros(n, 1);
LDA = zeros(1,3);
% 2. Calculate the LDA equation
for i = 1:n
    for j = 1:numofClass
        LDA(1,j) = mean(j,:) * inv(cov(:,:)) * X_test(i,:)' - 0.5 * mean(j,:) * inv(cov(:,:)) * mean(j,:)' + log(p(j));
    end
    % Find the argmax of LDA
    [~, argmax] = max(LDA);
    Y_predict(i,1) = argmax;
end

end
