load('data_knnSimulation.mat')

%gscatter(Xtrain(:,1),Xtrain(:,2),ytrain,'rgb','xo*')

% Split data to different calss
ind1 = ytrain == 1;
ind2 = ytrain == 2;
ind3 = ytrain == 3;
Xclass1 = Xtrain(ind1,:);
Xclass2 = Xtrain(ind2,:);
Xclass3 = Xtrain(ind3,:);

test = Xclass1;
data = Xtrain;
labels = ytrain;
k = 10;
[datarow , ~] = size(data);
diffMat = repmat(test,[datarow,1]) - data ;
distanceMat = sqrt(sum(diffMat.^2,2));
[B , IX] = sort(distanceMat,'ascend');
len = min(k,length(B));
relustLabel = mode(labels(IX(1:len)));

x = -3.5:0.1:6;
y = -3: 0.1:6.5;

clev = [-5 -3 -2:.5:2 3 5];
contourf(x,y,Xtrain,clev);