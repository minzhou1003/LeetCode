function [Y_predict] = minzhou_QDA_test(X_test, QDAmodel, numofClass)
%
% Testing for QDA
%
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% QDAmodel: the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance
% matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
% 
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test
% 1. Get the training parameters
n = size(X_test, 1);
Mean = QDAmodel.Mu;
covar = QDAmodel.Sigma;
p = QDAmodel.Pi;

Y_predict = zeros(n, 1);
QDA = zeros(1,3);
% 2. Calculate the QDA equation
for i = 1:n
    for j = 1:numofClass
        QDA(1,j) = 0.5 * ((X_test(i,:) - Mean(j,:))/covar(:,:,j) * (X_test(i,:) - Mean(j,:))') + 0.5 * log(det(covar(:,:,j))) - log(p(j));
    end
    % Find the argmin of QDA
    [~, argmin] = min(QDA);
    Y_predict(i,1) = argmin;
end

end
