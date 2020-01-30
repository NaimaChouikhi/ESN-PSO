function [AO, ACT, AOP] = rnn_esn_sim(net, IP, TP, tfsteps, N)
% RNN_ESN_SIM - simulates ESN
% [AO, ACT] = rnn_esn_sim(net, IP, TP, tfsteps, N, PSTEP)
% AO  - activities of output units (delay acts. are removed)
% ACT - activities of all units (inp. hidn. outp.), incl. init. acts.
% AOP - activities of all units 
% net - ESN network
% IP  - input sequence
% TP  - desired sequence
% tfsteps - - initial "teacher-forced" steps
% N   - noise to add

% get size of input sequence
[sizeInputPatt, numInputPatt] = size(IP);
if sizeInputPatt ~= net.numInputUnits; error ('Number of input units and input patterns do not match.'); end;

% get size of output sequence
[sizeOutputPatt, numOutputPatt] = size(TP);
if sizeOutputPatt ~= net.numOutputUnits; error ('Number of output units and output patterns do not match.'); end;

% calculate no. of teacher forced steps
if tfsteps==0, tfsteps = numOutputPatt; end;
if tfsteps>numOutputPatt, error('Number of teacher forced steps is too high.'); end;

% calculate starting and stopping step
firstStep = net.maxDelay+1;
lastStep  = net.maxDelay+numInputPatt;

% prepare activities (threshod + all input, then initial hidden and output)
ACT = zeros(net.numAllUnits, lastStep);
ACT(1:net.numAllUnits, 1:net.maxDelay) = net.actInit;
ACT(1,:) = 1;
ACT(2:net.numInputUnits+1,firstStep:lastStep) = IP;
TP = [net.actInit(net.numAllUnits-net.numOutputUnits+1:net.numAllUnits,:) ,TP];
AOP = zeros(net.numOutputUnits, lastStep);

% get unit counts
AUC = net.numAllUnits;
IUC = net.numInputUnits;
OUC = net.numOutputUnits;

% copy params (Matlab 13 Acceleration)
% add ending destination to unused value -1
numWeights = net.numWeights;
weightsDest   = [net.weights.dest]; weightsDest(end+1) = -1;
weightsSource = [net.weights.source];
weightsDelay  = [net.weights.delay];
weightsValue  = [net.weights.value];
unitsActFunc  = [net.units.actFunc];
unitsActFuncC1= [net.units.actFuncC1];
unitsActFuncC2= [net.units.actFuncC2];

% forward computation
for SI=(firstStep:lastStep),
    % initial settings
    nextdest = weightsDest(1);
    WI = 1;
    while WI<=numWeights,
        % next activity and initial destinantion node  
        act = 0;
        dest=nextdest;
        while dest==nextdest,
            % calculation
            source = weightsSource(WI);
            if (source <= AUC-OUC) || (SI-firstStep+1 > tfsteps),
                act = act + weightsValue(WI) .* ACT(weightsSource(WI), SI-weightsDelay(WI));
            else
                act = act + weightsValue(WI) .*  TP(weightsSource(WI)-AUC+OUC, SI-weightsDelay(WI));
            end;
            
            % get next destination node
            WI = WI+1;
            nextdest = weightsDest(WI);
        end;

        % calculate activity
        if unitsActFunc(dest) == 1, ACT(dest, SI) = unitsActFuncC1(dest)*ACT(dest, SI-1) + unitsActFuncC2(dest)*tanh(act); 
        elseif unitsActFunc(dest) == 2, ACT(dest, SI) = unitsActFuncC1(dest)*ACT(dest, SI-1) + unitsActFuncC2(dest)*act; 
        else error('Unknown act. function');
        end;
    end;
    
    % add random noise to hidden only activities
    if N ~= 0.0, ACT(IUC+1:AUC-OUC, SI) = ACT(IUC+1:AUC-OUC, SI) + 2*(rand(AUC-OUC-IUC, 1)-0.5) .* N; end;
end;    

% select output activities
AO = ACT(net.indexOutputUnits, firstStep:lastStep);
