% training_esn_mg_pred - training ESN on Mackey Glass time series, for prediction signal STEP values ahead
clear all
close all
clc
% seed random generator
tic;
rand('state', sum(100*clock));

% prediction STEP
STEP = 6;

% unit counts (input, hidden, output)
IUC = 1;
HUC = 50;
OUC = 1;

% initialize ESN weights

probInp  = [  1.00 ];
rngInp   = [  1.00 ]; 
probRec  = [  0.1 ];
rngRec   = [  -0.6 ];
probBack = [  0.1];
rngBack  = [  0.1 ];

% create input and output time series with delay=TAU
TAU = 17; 
if ~exist('MGS');
    MGS = tanh(createmgdde23(3000, TAU, 200)-1);
end;
x=MGS./10;
IP=x(1:500-STEP);
TP=x(1+STEP:500);
IPT=x(501:1000-STEP);
TPT=x(501+STEP:1000);
IP1=x(1001:1500-STEP);
TP1=x(1001+STEP:1500);

Lambda = 0.0;
UnitAct = 11;
[net] = rnn_esn_new(IP, TP, IP1, TP1, IPT, TPT,IUC, HUC, OUC, probInp, rngInp, probRec, rngRec, probBack, rngBack, Lambda, UnitAct);
%[net, old_max_eig] = rnn_ffesn_new(IUC, HUC, OUC, probInp, rngInp, probRec, rngRec, probBack, rngBack, 0.0, 0.75);
% fprintf('Maximal eig. is (before scaling) %f\n', old_max_eig);

% train network using 3000 values of MG seq. (ev. add noise to the target seq.)

[net, MSE] = rnn_esn_train(net, IP, TP, 50, 0.0);
fprintf('Training RMSE after PSO pre-training is %g\n', MSE);

% test network using 2000 values of MG seq. as (teacher-forced) initial sequence

[AO, ACT] = rnn_esn_sim(net, IPT, TPT,0, 0.0);
MSE = eval_mse(AO, TPT);
fprintf('Testing RMSE after PSO pre-training is %g\n', MSE);
toc
% plot results
TPT=TPT.*10;
AO=AO.*10;
figure(1)
plot(TPT,'r'); hold on;
plot(AO,'b-.'); hold off;
title('Desired outputs vs network outputs')
saveas(gcf, '../results/fig1.png')
print('../results/plot', '-dpdf')

