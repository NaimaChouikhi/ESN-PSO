function MGS = createmg(LEN, SUBS, ALPHA, BETA, GAMA, TAU, INITDUMMY)
% create Mackey-GLass time series
% LEN   - sequence length
% SUBS  - subsampling
% ALPHA, BETA, GAMA, TAU - Mackey-GLass time series parameters
% INITDUMMY - initial steps to supress (subsampling not taken into account)

% set default values
if nargin < 2, SUBS = 10; end;
if nargin < 3, ALPHA = 0.2; end;
if nargin < 4, BETA = 10; end;
if nargin < 5, GAMA = 0.1; end;
if nargin < 6, TAU = 17; end;
if nargin < 7, INITDUMMY = 1000; end;

% set positions
start = SUBS * TAU + 1;
stop  = SUBS * (INITDUMMY + TAU + LEN + 1);
delay = SUBS * TAU;

% preallocate array and set up initial conditions
S = zeros(1,  stop);
S(start) = 1;

% create Mackey-Glass time series
for SI=(start:stop-1),
    NOM1 = ALPHA * S(SI-delay);
    DEN1 = 1 + S(SI-delay)^BETA;
    S(SI+1) = S(SI) + (NOM1 / DEN1 - GAMA * S(SI)) / SUBS;
end;

% remove dummy steps and take samples into output sequence
S(1:SUBS * (INITDUMMY + TAU)) = [];
MGS = S(1:SUBS:SUBS*LEN);