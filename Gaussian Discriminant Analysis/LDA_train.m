function [LDAmodel] = minzhou_LDA_train(X_train, Y_train, numofClass)
%
% Training LDA
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
% LDAmodel : the parameters of LDA classifier which has the following fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% **************************************
% Split data to different calss
ind1 = Y_train == 1;
ind2 = Y_train == 2;
ind3 = Y_train == 3;
Xclass1 = X_train(ind1,:);
Xclass2 = X_train(ind2,:);
Xclass3 = X_train(ind3,:);
% **************************************
% Calculate the Pi and Mu
Pi = [size(Xclass1,1); size(Xclass2,1);size(Xclass3,1)]/size(X_train,1);
Mu = [mean(Xclass1); mean(Xclass2); mean(Xclass3)];
% **************************************
% Calculate the Sigma for LDA
Sigma = cov(Xclass1)*Pi(1) + cov(Xclass1)*Pi(2) + cov(Xclass3)*Pi(3);
            
LDAmodel.Mu = Mu;
LDAmodel.Sigma = Sigma;
LDAmodel.Pi = Pi;

end
