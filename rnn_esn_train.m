function [neto, MSE, AO, ACT] = rnn_esn_train(net, IP, TP, dsteps, N)
% RNN_ESN_TRAIN - ESN training
% [neto, MSE, AO, ACT] = rnn_esn_train(net, IS, TS, dsteps, noise, alpha, noforwardfrominput)
% neto- trained ESN network
% MSE - training error
% AO  - activities of output units (delay acts. are removed)
% ACT - activities of all units (inp. hidn. outp.), incl. init. acts.
% net - ESN network
% IP  - input sequence
% TP  - desired sequence
% dsteps - initial "dummy" steps
% N   - noise to add


% get size of input sequence
[sizeInputPatt, numInputPatt] = size(IP);
if sizeInputPatt ~= net.numInputUnits; error ('Number of input units and input patterns do not match.'); end;

% get size of output sequence
[sizeOutputPatt, numOutputPatt] = size(TP);
if sizeOutputPatt ~= net.numOutputUnits; error ('Number of output units and output patterns do not match.'); end;

% check length of input
if numInputPatt ~= numOutputPatt, error('Number of input units and input patterns do not match.'); end;

% get values
AUC = net.numAllUnits;
IUC = net.numInputUnits;
OUC = net.numOutputUnits;

maxDelay = net.maxDelay;
firstForwardWeight = net.firstForwardWeight;

% delete forward weights
net.numWeights = firstForwardWeight-1;
net.weights(firstForwardWeight:end) = [];

% copy params (Matlab 13 Acceleration)
unitsActFunc  = [net.units.actFunc];
unitsActFuncC1= [net.units.actFuncC1];
unitsActFuncC2= [net.units.actFuncC2];

% simulate ESN with TP
[AO, ACT] = rnn_esn_sim(net, IP, TP, 0, N);

% throw out dummy activities
ACT = ACT(1:AUC-OUC, maxDelay+dsteps+1:end);


% get net outputs by act. f.
outUnitsActFunc = unitsActFunc(AUC-OUC+1:AUC);
DAO_ATANH = atanh(TP(:, dsteps+1:end)); 
DAO_LIN = TP(:, dsteps+1:end); 
DAO(outUnitsActFunc==1,:) = DAO_ATANH(outUnitsActFunc==1,:);
DAO(outUnitsActFunc==2,:) = DAO_LIN(outUnitsActFunc==2,:);

% fit forward weights 
weightsForward =  DAO*pinv(ACT);

% fill forward weights into net
nw = firstForwardWeight;
for i=(1:OUC),
    % input weights
    for j=(1:IUC+1),
        value = weightsForward(i,j);
        if value ~= 0,
            net.weights(nw).value  = value;
            net.weights(nw).dest   = i+AUC-OUC;
            net.weights(nw).source = j;
            net.weights(nw).delay  = 0;
            nw = nw+1;
        end
    end;
    
    % recurrent weights
    for j=(IUC+2:AUC-OUC),
        value = weightsForward(i,j);
        if value ~= 0,
            net.weights(nw).value  = value;
            net.weights(nw).dest   = i+AUC-OUC;
            net.weights(nw).source = j;
            net.weights(nw).delay  = 0;
            nw = nw+1;
        end;
    end;
end;        

% set number of weights
net.numWeights = nw-1;

% simulate trained ESN with TP
if nargout>1,
    [AO, ACT] = rnn_esn_sim(net, IP, TP, 0, 0);
    MSE = eval_mse( AO(:,dsteps+1:end), TP(:,dsteps+1:end) );
end;

% create output ESN object
neto = net;


