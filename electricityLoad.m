%plot(PETPRISPTS1MS1(:,2));
load data.mat;
%5 day of testing
[n,~] = size(ELECTRICITY_LOAD);
testSize = 5*24*60;
trainingSize = n-testSize;
%global activity is col 4, kitchen load is col 8
col = 8;

trainingSet = ELECTRICITY_LOAD(1:trainingSize,col);
testSet = ELECTRICITY_LOAD(trainingSize+1:n,col);
%% define parameter
% trainingSetStat = filter(D2,trainingSet);
% figure;
% autocorr(trainingSetStat);
% figure;
% parcorr(trainingSetStat);
%% build model
minV = Inf;
MSE=zeros(4,4,4);
for r =0:3
    for i=0:3
        for m=0:3
            modelParam = arima(r,i,m);
            model = estimate(modelParam,trainingSet);
            %test on data set
            var = infer(model,testSet);
            MSE(r+1,i+1,m+1)= sum(var.^2)/testSize;
            if(MSE(r+1,i+1,m+1)<minV)
                bestModel = model;
            end
        end
    end
end
%%
% %forecast
% trueVal = filter(D2,testSet);
% [pred,varPred] = forecast(model,5);
% pred = pred(1:3);
% varPred = varPred(1:3);
% err = trueVal(1:3) - pred;
% err'*err
% 
% figure;
% plot(trueVal);
% hold on;
% errorbar(pred,varPred);
%%
% pred = zeros(1,n);
% varPred = zeros(1,n);
% for i=2:n
%     [pred(i),varPred(i)] = forecast(model,1,'Y0',ELe(i-1,2));
% end
% %testSet = filter(D2,testSet);
% figure;
% %errorbar(pred(2:n),varPred(2:n));
% plot(pred(2:n));
% hold on;
% plot([trainingSet;testSet]);
%%
t1 = datetime(2007,1,1,0,0,0);
t2 = datetime(2007,12,31,23,59,0);
timeLabel = t1:minutes(1):t2;
%% fitted error
a = 15000; 
b = 30000;
residualTR = infer(bestModel,trainingSet);
figure;
plot(trainingSet(a:b),'-b');
hold on;
plot(trainingSet(a:b) - residualTR(a:b),'-r');
title('Training Set');
figure;
plot(abs(residualTR(a:b)),'-g');
title('Residual Training Set');
%% preditcion of test set
residualTS = infer(bestModel,testSet);
figure;
plot(timeLabel(trainingSize+1:n),testSet,'-b');
hold on;
plot(timeLabel(trainingSize+1:n),testSet - residualTS,'-r');
title('ARIMA(3,0,2)');
legend('True Value','Forecasted Value');