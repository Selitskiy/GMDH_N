%% Clear everything 
clearvars -global;
clear all; close all; clc;

%% Mem cleanup
ngpu = gpuDeviceCount();
for i=1:ngpu
    reset(gpuDevice(i));
end

%% Load the data, initialize partition pareameters
%saveDataPrefix = 'nasdaq3520_';
%saveDataPrefix = 'dj4020_';
saveDataPrefix = 'nikkey4030_';
%saveDataPrefix = 'AirPassengers1_114_30_';
%saveDataPrefix = 'sun_1_';
%saveDataPrefix = 'SN_y_tot_V2.0_spots_4030_';

save_identNet_fileT = '~/data/ws_van_ident_';
save_regNet_fileT = '~/data/ws_van_reg_';

%dataFile = 'nasdaq_1_3_05-1_28_22.csv';%'./wse_data.csv';
%dataFile = 'dj_1_3_05-1_28_22.csv';
dataFile = 'nikkei_1_4_05_1_31_22.csv';

%dataFile = 'AirPassengers1.csv';
%dataFile = 'sun_1.csv';
%dataFile = 'SN_y_tot_V2.0_spots.csv';


Me = readmatrix(dataFile);
[l_whole_ex, ~] = size(Me);


% input dimesion (days)
m_in = 30;
% Try different output dimensions (days)
n_out = 30;

% Allocate place for future
%M = zeros([l_whole_ex+n_out, 1]);
%M(1:l_whole_ex) = Me;
% Leave space for last training label
%l_whole = l_whole_ex-m_in;

% Or no future
M = Me;
% Leave space for last training label
l_whole = l_whole_ex-m_in-n_out;

% Break the whole dataset in training sessions,
% set training session length
l_sess = 2*m_in;

% Only for 1 whole session (otherwise, comment out)
%l_sess = l_whole;

% No training n_out touches test m_in
n_sess = floor((l_whole)/l_sess);%floor((l_whole-l_test)/l_sess);


% Normalization flag
norm_fl = 1;
    
% Reformat sequence into input training tensor X of m_in observations dimension, 
% label Y (predicted values) of n_out dimension, on training seesion of
% length l_sess (n_sess of them), and min-max boundaries B for each observation, 
% such that number of observations in a session, including training label sequences
% do not touch test period
[X, Y, B, XI, C, k_ob, m_ine, n_oute] = w_seriesva_train_tensors(M, m_in, n_out, l_sess, n_sess, norm_fl);
n_outv = n_oute - n_out;


% Fit ann into minimal loss function (SSE)
mult = 1;
k_hid = floor(mult * m_ine);
k_hid1 = floor(mult * (m_ine + 1));
k_hid2 = floor(mult * (2*m_ine + 1));
identNets = cell(n_sess);
regNets = cell(n_sess);

max_neuro1 = floor(k_hid1 / n_out);
max_neuro2 = floor(k_hid2 / n_out);

%% Attention Input Identity net
% Train or pre-load Identity nets

for i = 1:n_sess

    mb_size = 2^floor(log2(k_ob*i)); %32

    aLayers = [
    featureInputLayer(m_ine)
    fullyConnectedLayer(k_hid1)
    reluLayer
    fullyConnectedLayer(k_hid2)
    reluLayer
    fullyConnectedLayer(n_sess)
    softmaxLayer
    classificationLayer
    ];
    agraph = layerGraph(aLayers);

    sOptionsAttention = trainingOptions('adam', ...
            'ExecutionEnvironment','parallel',...
            'Shuffle', 'every-epoch',...
            'MiniBatchSize', mb_size, ...
            'InitialLearnRate',0.02, ...
            'MaxEpochs',250);


    save_identNet_file = strcat(save_identNet_fileT, saveDataPrefix, int2str(i), '_', int2str(m_in), '_', int2str(n_out), '_', int2str(n_sess), '.mat');
    if isfile(save_identNet_file)
        fprintf('Loading Ident net %d from %s\n', i, save_identNet_file);
        load(save_identNet_file, 'identNet');
    else
        clear('identNet');
    end

    if exist('identNet') == 0
        fprintf('Training Ident net %d\n', i);

        identNet = trainNetwork(XI(:, 1:k_ob*i)', C(1:k_ob*i)', agraph, sOptionsAttention);
        
        save(save_identNet_file, 'identNet');
    end

    identNets{i} = identNet;
end


%% GMDH parameters 

mb_size = 2^floor(log2(k_ob)); %32

sOptions = trainingOptions('adam', ...
'ExecutionEnvironment','parallel',...
'Shuffle', 'every-epoch',...
'MiniBatchSize', mb_size, ...
'InitialLearnRate',0.02, ...
'MaxEpochs',500);%, ...
            %'Verbose',true, ...
            %'Plots','training-progress');
    
            %'LearnRateSchedule', 'piecewise',...
            %'LearnRateDropPeriod', 100,...
            %'LearnRateDropFactor', 0.99,...

sOptions2 = trainingOptions('adam', ...
'ExecutionEnvironment','parallel',...
'Shuffle', 'every-epoch',...
'MiniBatchSize', mb_size, ...
'InitialLearnRate',0.03, ...
'MaxEpochs',1000);

 sOptionsFinal = trainingOptions('adam', ...
'ExecutionEnvironment','parallel',...
'Shuffle', 'every-epoch',...
'MiniBatchSize', mb_size, ...
'InitialLearnRate',0.01, ...
'MaxEpochs',250);           

%% Train or pre-load GMDH
for i = 1:n_sess

    save_regNet_file = strcat(save_regNet_fileT, saveDataPrefix, int2str(i), '_', int2str(m_in), '_', int2str(n_out), '_', int2str(n_sess), '.mat');
    if isfile(save_regNet_file)
        fprintf('Loading net %d from %s\n', i, save_regNet_file);
        load(save_regNet_file, 'regNet');
    else
        clear('regNet');
    end


    if exist('regNet') == 0

    % Start growing GMDH net
    prevLayerName = 'inputFeature';
    oLayers = [
        featureInputLayer(m_ine, 'Name', prevLayerName)
    ];
    cgraph = layerGraph(oLayers);


    fprintf('Training net %d\n', i);

    % Start from first GMDH layer
    ll = 1;
    % Target accuracy
    accTarget = 0.4;
    % Stale accuracy chage threshol
    dAccMin = 0.001;
    % Maximal polinomial length (number of chained gmdh layer) 
    lMax = 3;

    [cgraph, regNet, ll, k_hid1_real, curGMDHLayerName, curGMDHRegressionName] =... 
        gmdhLayerGrowN(cgraph, prevLayerName, sOptions, X, Y, m_ine, i, n_oute, ll, accTarget, max_neuro1, 0, dAccMin, lMax);
    prevLayerName = curGMDHLayerName;

    if(k_hid1_real > 1)
        % Sum all polynomial candidates into last fully connected layuer and use
        % standard Regression instead of GMDH
        fullyConnectdMidLayerName = 'fullyConnectedLayerMid';
        fullyConnectdMidLayer = fullyConnectedLayer(k_hid1_real, 'Name', fullyConnectdMidLayerName);
        cgraph = addLayers(cgraph, fullyConnectdMidLayer);
        cgraph = connectLayers(cgraph, curGMDHLayerName, fullyConnectdMidLayerName);

        prevLayerName = fullyConnectdMidLayerName;


        fprintf('Training 2 GMDH net %d\n', i);

        % Build second GMDH layer
        ll = ll + 1;
        % Target accuracy
        accTarget = 0.3; %0.015;% 0.05; 0.08;
        % Stale accuracy chage threshol
        dAccMin = 0.001;
        % Maximal polinomial length (number of chained gmdh layer) 
        lMax = 3;

        [cgraph, regNet, ll, k_hid2_real, curGMDHLayerName, curGMDHRegressionName] =... 
            gmdhLayerGrowN(cgraph, prevLayerName, sOptions2, X, Y, k_hid1_real, i, n_oute, ll, accTarget, max_neuro2, 0, dAccMin, lMax);
        prevLayerName = curGMDHLayerName;

        if(k_hid2_real > 1)
            fullyConnectdLastLayerName = 'fullyConnectedLayerLast';
            fullyConnectdLastLayer = fullyConnectedLayer(k_hid2_real, 'Name', fullyConnectdLastLayerName);
            cgraph = addLayers(cgraph, fullyConnectdLastLayer);
            cgraph = connectLayers(cgraph, curGMDHLayerName, fullyConnectdLastLayerName);

            fullyConnectedOutLayerName = 'fcOut';
            fullyConnectedOutLayer = fullyConnectedLayer(n_oute,'Name',fullyConnectedOutLayerName);
            cgraph = addLayers(cgraph, fullyConnectedOutLayer);
            cgraph = connectLayers(cgraph, fullyConnectdLastLayerName, fullyConnectedOutLayerName);

            prevLayerName = fullyConnectedOutLayerName;
        end
    end

    regressionLayerName = "regOut";
    %regressionLayer = regressionLayer('Name', regressionLayerName);
    regressionLayer = vRegression(regressionLayerName, n_outv);

    cgraph = replaceLayer(cgraph, curGMDHRegressionName, regressionLayer);
    cgraph = connectLayers(cgraph, prevLayerName, regressionLayerName);


    fprintf('Training whole GMDH net %d\n', i);

    regNet = trainNetwork(X(:,:,i)', Y(:,:,i)', cgraph, sOptionsFinal);

    save(save_regNet_file, 'regNet');
    end

    regNets{i} = regNet;

    clear('regressionLayerName');
    clear('regressionLayer');
    clear('cgraph');
    clear('regNet');

end

%% Test parameters 
% the test input period
l_test = m_in;

% Test from particular training session
sess_off = 0;
% additional offset after training sessions (usually for the future forecast)
offset = 0;

% Left display margin
l_marg = 1;

%% Test parameters for one last session

% Left display margin
%l_marg = 4100;

% Future session
%M = zeros([l_whole_ex+n_out, 1]);
%M(1:l_whole_ex) = Me;
%[l_whole, ~] = size(M);

% Last current session
%l_whole = l_whole_ex;

% Fit last testing session at the end of data
%offset = l_whole - n_sess*l_sess - m_in - n_out;

% Test from particular training session
%sess_off = n_sess-1;

%% For whole-through test, comment out secion above

[X2, Y2, Yh2, Bt, k_tob] = w_seriesva_test_tensors(M, m_in, n_out, l_sess, l_test, n_sess, sess_off, offset, norm_fl, m_ine, n_oute, 0);


%% test
%i = 1;
for i = 1:n_sess-sess_off
    for k = 1:k_tob

        predictClass = classify(identNets{i+sess_off}, X2(:, k, i)');
        prClNum = double(predictClass);
        fprintf('IdentityClass Session:%d, Observation:%d, IdentClNum:%d\n', i, k, prClNum);

        predictedScores = predict(regNets{prClNum}, X2(:, k, i)');
        Y2(:, k, i) = predictedScores';

    end
end

    
%% re-scale in observation bounds
if(norm_fl)
    for i = 1:n_sess-sess_off
        for j = 1:k_tob
            Y2(:, j, i) = w_series2_rescale(Y2(:, j, i), Bt(1,j,i), Bt(2,j,i));
            Yh2(:, j, i) = w_series2_rescale(Yh2(:, j, i), Bt(1,j,i), Bt(2,j,i));
        end
    end
end


%% Calculate errors
[S2, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = w_seriesv_calc_err(Y2, Yh2, n_out); 

fprintf('GMDH ANN M in:%d, N out:%d, Sess:%d ,Err: %f\n', m_in, n_out, n_sess, S2);

%% Error and Series Plot
%w_series2_err_graph(Y2, Yh2);
w_seriesva_ser_graph(M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, n_sess, sess_off, offset, l_marg);

