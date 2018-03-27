function [RDAmodel]= minzhou_RDA_train(X_train, Y_train,gamma, numofClass)
%
% Training RDA
%
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_train : training data matrix, each row is a training data point
% Y_train : training labels for rows of X_train
% numofClass : number of classes 
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
% **************************************
% Split data to different calss
ind0 = Y_train == 0;
ind1 = Y_train == 1;
Xclass0 = X_train(ind0,:);
Xclass1 = X_train(ind1,:);
% % **************************************
% Calculate the Pi and Mu
Pi = [size(Xclass0,1); size(Xclass1,1)]/size(X_train,1);
Mu = [mean(Xclass0); mean(Xclass1)];
% **************************************
% Calculate the Sigma for LDA
Sigma = cov(Xclass0)*Pi(1) + cov(Xclass1)*Pi(2);
% RDA calculate Sigmapooled
cov_diag_ele = diag(Sigma);
cov_diag = diag(cov_diag_ele);
% Regularized the covariance matrix
Sigmapooled = gamma' * cov_diag + (ones(length(gamma),1) - gamma')* Sigma;

RDAmodel.Mu = Mu;
RDAmodel.Sigmapooled = Sigmapooled;
RDAmodel.Pi = Pi;

end
