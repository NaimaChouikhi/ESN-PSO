function [neto, oldSpecRad] = rnn_esn_new(IP, TP, IP1, TP1, IPT, TPT, IUC, HUC, OUC, probInp, rngInp, probRec, rngRec, probBack, rngBack, specRad, unitAct)
% [neto, oldSpecRad] = rnn_esn_new(IUC, HUC, OUC, probInp, rngInp, probRec, rngRec, probBack, rngBack, specRad, unitAct)
% neto - new network structure
% oldSpecRad - old spectral radius of recurrent weight matrix
% IUC - number of input units
% HUC - number of hidden units
% OUC - number of output units
% probInp  - vector of prob. of input weights
% rngInp   - vector of input weights' ranges
% probRec  - vector of prob. of recurrent weights
% rngRec   - vector of recurrent weights' ranges
% probBack - vector of prob. of backward weights
% rngBack  - vector of backward weights' ranges
% specRad  - required spectral radius (if omited or zero -> no scaling)
% unitAct  - units' activation function specifier (0 - tanh, 1 - lin. for hid. units

% set number of all units
AUC = 1+IUC+HUC+OUC;

% set numbers of units
net.numInputUnits    = IUC;
net.numHiddenUnits   = HUC;
net.numOutputUnits   = OUC;
net.numAllUnits      = AUC;

% set neuron masks
net.maskInputUnits   = [0; ones(IUC, 1); zeros(AUC-1-IUC, 1)];
net.maskOutputUnits  = [zeros(AUC-OUC, 1); ones(OUC, 1)];
net.indexOutputUnits = find(net.maskOutputUnits);
net.indexInputUnits  = find(net.maskInputUnits);


% set weight matrices
inputWeights = zeros(HUC, IUC+1, length(probInp));
recurrentWeights = zeros(HUC, HUC, length(probRec));
backwardWeights = zeros(HUC, OUC, length(probBack));

for d=(1:length(probInp))
    inputWeights(:,:,d) = init_weights(inputWeights(:,:,d), probInp(d),rngInp(d));
end;

for d=(1:length(probRec))
    recurrentWeights(:,:,d) = init_weights(recurrentWeights(:,:,d), probRec(d),rngRec(d));
end;

for d=(1:length(probBack))
    backwardWeights(:,:,d) = init_weights(backwardWeights(:,:,d), probBack(d),rngBack(d));
end;

    
% init parameters
if nargin<10, specRad = 0.0; end;


% scale to defined spectral radius
oldSpecRad = NaN;
if (nargout>1 || specRad>0) && length(probRec)>=1,
    oldSpecRad = max(abs(eig(recurrentWeights(:,:,1))));
end;

if specRad>0,
    recurrentWeights = recurrentWeights ./ oldSpecRad .* specRad;
end;

% init parameters
if nargin<11, unitAct = 0; end;

% set units
unit = struct('actFunc',1,'actFuncC1',0.0,'actFuncC2',1.0);
for i=(1:AUC), net.units(i) = unit; end;
if unitAct==21, for i=(IUC+2:AUC-OUC), net.units(i).actFunc = 2; end; 
elseif unitAct==22; for i=(IUC+2:AUC), net.units(i).actFunc = 2; end;
elseif unitAct==12; for i=(AUC-OUC+1:AUC), net.units(i).actFunc = 2; end;
elseif unitAct==0;
elseif unitAct==11;
else error('Unknown unit activation function specifier.'); 
end;

% set weights
weight = struct('dest',0,'source',0,'delay',0,'value',0);

nw = 1;
for i=(IUC+2:AUC-OUC),
    % input weights
    for j=(1:IUC+1),
        for d=(1:length(probInp)),
            value = inputWeights(i-IUC-1, j, d);%init_weight(probInp(d),rngInp(d));
            if value ~= 0,
                net.weights(nw).value  = value;
                net.weights(nw).dest   = i;
                net.weights(nw).source = j;
                net.weights(nw).delay  = d-1;
                nw = nw+1;
            end;
        end;
    end;
end
k1=nw;

Recurrent=[];
for i=(IUC+2:AUC-OUC),
    % recurrent weights
    for j=(IUC+2:AUC-OUC),
        for d=(1:length(probRec)),
            value = recurrentWeights(i-IUC-1, j-IUC-1, d);
            if value ~= 0,
                net.weights(nw).value  = value;
                net.weights(nw).dest   = i;
                net.weights(nw).source = j;
                net.weights(nw).delay  = d;
                nw = nw+1;
                Recurrent=[Recurrent value];
            end;
        end;
    end;
end
%[m1,n1]=size(Recurrent)
k2=nw;

for i=(IUC+2:AUC-OUC),
    % backward weights
    for j=(AUC-OUC+1:AUC),
        for d=(1:length(probBack)),
            value = backwardWeights(i-IUC-1, j-AUC+OUC);
            if value ~= 0,
                net.weights(nw).value  = value;
                net.weights(nw).dest   = i;
                net.weights(nw).source = j;
                net.weights(nw).delay  = d;
                nw = nw+1;
            end;
        end;
    end;
end;
k3=nw;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Copy the values of the structure %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     weights in another matrix     %%%%%%%%%%%
weightsValue  = [net.weights.value];
[m2,n2]=size(weightsValue);

partWeights=[];
b=-40;
c=3;
 for (i=k1+b:k1+(k2-k1-1)+c)
   partWeights=[partWeights weightsValue(i)];
 end
 [m1,n1]=size(partWeights);

%end
% set number of weights
net.numWeights = nw-1;
net.firstForwardWeight = nw;

    
% initialize starting activities from [0, 1]
net.maxDelay = max([length(probInp)-1, length(probRec), length(probBack)]);

% initialize initial activations from [-1, 1]
net.actInit = zeros(net.numAllUnits, net.maxDelay);
% net.actInit = 2.0 * rand(net.numAllUnits, net.maxDelay) - 1.0;
[net, MSE, AO, ACT] = rnn_esn_train(net, IP, TP,50, 0.0);
fprintf('Training RMSE before PSO pre-training is %g\n', MSE);

% test network using 250 values of MG seq. as initial (teacher-forced) sequence
[AO, ACT] = rnn_esn_sim(net, IPT, TPT,0, 0.0);
MSE = eval_mse(AO, TPT);
fprintf('Testing RMSE before PSO pre-training is %g\n', MSE)
dsteps=50;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Training  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   database   %%%%%%%%%%%%%%%%%%%%%%%%%%%%


%neto = net;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         Exceptions            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   if dimensions don't match  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         PSO            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Pre-Training          %%%%%%%%%%%%%%%%%%%%%%%%%%%%


n = floor((AUC-(IUC+OUC))/8)  ;      % Size of the swarm " no of birds "
bird_setp  =50; % Maximum number of "birds steps"
c2 =2.2;          % PSO parameter C1 
c1 =0.12;        % PSO parameter C2 
w =0.9;          % PSO parameter w
R1 = rand(n1, n);   % PSO randomness parameter R1
R2 = rand(n1, n);   % PSO randomness parameter R2
current_fitness =0*ones(n,1); % initialize current fitness
for (i=1:n)
     current_position(:,i)=partWeights ;   % particles positions initialization from reservoir weights (equal in the beginning) 
     %current_position(:,i)=0.4-0.8*rand(n1, 1);
end
vmax=2;
vmin=-2;
% current_position = (rand(n1, n)-.5);
velocity=0.2*randn(n1, n) ;     % particles velocities initialization
local_best_position  = current_position; % particles local best positions initialization
size(velocity);
 for i = 1:n
    current_fitness(i) = MSE;    % random fitness initialization
 end
local_best_fitness  = current_fitness ;  % local best fitness initialization
[global_best_fitness,g] = min(local_best_fitness); % global best fitness initialization
for i=1:n
    globl_best_position(:,i) = local_best_position(:,g) ;  % global best position initialization
end
velocity = w *velocity + c1*(R1.*(local_best_position-current_position)) + c2*(R2.*(globl_best_position-current_position)); % particles current velocities initialization according to PSO formula
current_position = current_position + velocity ;   % particles current positions initialization according to PSO formula
%pmax=1.00;
%pmin=-1.00;
iter = 0 ;          % Iterations counter
glblbest_fitness=[];
while  ( iter < bird_setp )
iter = iter + 1                            % Iterations counter increment
fitness=[];                                 % fitness vector intialtization
                        % global best  fitness vector intialtization

NulVector=zeros(1,nw);                     % creating a zero vector of length of net.weight.value 
NulVector2net=num2cell(NulVector);         % transforming the elements of the vector into cells
[net.weights.value]=NulVector2net{:};      %assigning these cells values to net.weights.value
net.weights.value;
 [m6,n6]=size(weightsValue);
 [m7,n7]=size(current_position(:,1));

for i = 1:n,
    weightsValue (k1+b:k1+(k2-k1-1)+c) = current_position(:,i)';  % injection of new values of current position into the reservoir net.weight.value 
    [m6,n6]=size(weightsValue);
    weightsValue2=[weightsValue zeros(1,nw-n6)];
     weightsValue2net=num2cell(weightsValue2);
    [net.weights.value] = weightsValue2net{:};               % injection of new values of current position into the reservoir net.weight.value
 
%  end

% simulate ESN with TP
[AO, ACT] = rnn_esn_sim(net, IP, TP, 0, 0);                 % simulate the network without considering the output weights values
%   ACT;
%   if i==1
%     A=ACT;
% end
% A-ACT
% pause
% throw out dummy activities

ACT = ACT(1:AUC-OUC, maxDelay+dsteps+1:end);                % extract the activities of all units except output units

%size(ACT)

% get net outputs by act. f.
outUnitsActFunc = unitsActFunc(AUC-OUC+1:AUC);
DAO_ATANH = atanh(TP(:, dsteps+1:end)); 
DAO_LIN = TP(:, dsteps+1:end); 
DAO(outUnitsActFunc==1,:) = DAO_ATANH(outUnitsActFunc==1,:);
DAO(outUnitsActFunc==2,:) = DAO_LIN(outUnitsActFunc==2,:);
%size(DAO)
%size(pinv(ACT))

% fit forward weights 
weightsForward = DAO*pinv(ACT);     % getting forward output weights

% fill forward weights into net
nw = firstForwardWeight;
for k=(1:OUC),
    % reservoir to outputs weights                                  

    for j=(1:IUC+1),                   % injection of the forward output weights values, sources, destinations
        value = weightsForward(k,j);
        if value ~= 0,
            net.weights(nw).value  = value;
            net.weights(nw).dest   = k+AUC-OUC;
            net.weights(nw).source = j;
            net.weights(nw).delay  = 0;
            nw = nw+1;
        end
    end;
    for j=(IUC+2:AUC-OUC),
        value = weightsForward(k,j);
        if value ~= 0,
            net.weights(nw).value  = value;
            net.weights(nw).dest   = k+AUC-OUC;
            net.weights(nw).source = j;
            net.weights(nw).delay  = 0;
            nw = nw+1;
        end;
    end;
end;        


% set number of weights
net.numWeights = nw-1;

% simulate trained ESN with TP
%if nargout>1,
    [AO, ACT] = rnn_esn_sim(net, IP, TP, 0, 0);       % simulation of the network after addition of output weight matrix
    MSE_Training = eval_mse( AO(:,dsteps+1:end), TP(:,dsteps+1:end)); % Mean Square error (fitness function) of each particle in this iteration
   % fitness=[fitness MSE_Training];
 
   [AO, ACT] = rnn_esn_sim(net, IP1, TP1, 0, 0);
    MSE_Test = eval_mse( AO, TP1);
    fitness=[fitness MSE_Test];                                   % Accumulation of Mean Square errors of all the particles in this iteration
    nw;
end;

current_fitness=fitness'                                    % Getting the matrix of the fitnesses
for i = 1 : n
        if current_fitness(i) < local_best_fitness(i)        % assigning the local best fitness into each particle
           local_best_fitness(i)  = current_fitness(i);  
           local_best_position(:,i) = current_position(:,i)   ;   % assigning the local best position into each particle
        end   
 end

  
 [current_global_best_fitness,g] = min(local_best_fitness);     % searching for the current global best fitness and its index
 current_global_best_fitness
  g
    
if current_global_best_fitness < global_best_fitness
   global_best_fitness = current_global_best_fitness;       % update the global best fitness 
    for i=1:n
        globl_best_position(:,i) = local_best_position(:,g);  % assigning the global best position into each particle
    end
   
end
glblbest_fitness=[glblbest_fitness global_best_fitness];
velocity = w *velocity + c1*(R1.*(local_best_position-current_position)) + c2*(R2.*(globl_best_position-current_position)); % velocities update to be used for the next iteration

current_position = current_position + velocity; % current positions update to be used for the next iteration
end                                             %end of iterations
 [Jbest_min,I] = min(current_fitness) ;    %   minimum fitness after the end of the iterations
 current_position(:,I);                    % its corresponding current position

 weightsValue (k1+b:k1+(k2-k1-1)+c) = current_position(:,I); % re-injection of new values of current position into the coreesponding reservoir net.weight.value
    [m6,n6]=size(weightsValue);
    weightsValue2=[weightsValue zeros(1,nw-n6)];
     weightsValue2net=num2cell(weightsValue2);
    [net.weights.value] = weightsValue2net{:};
    net.weights.value;
% create output ESN object
neto = net;

% initializing weights matrices

% initialze weight 
function weight = init_weight(prob, rng)

weight = 0;
if rand > prob, return; end;

weight = 2.0 * rand - 1.0;
if rng >= 0, 
    weight = weight .* rng;
else 
    if weight  < 0; weight =  rng;
    else weight = -rng; 
    end;
end; 



% initialze weights given as inputs
function weights = init_weights(weights, prob, rng)

mask = rand(size(weights)) < prob;
weights = (2.0 * rand(size(weights)) - 1.0) .* mask;
if rng >= 0, 
    weights = weights .* rng;
else 
    weights(weights < 0) = rng; 
    weights(weights > 0) = -rng;
end; 
