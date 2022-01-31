%% Clear everything 
clearvars -global;
clear all; close all; clc;
%cd('~/WCCI2');

%% Load the data, initialize partition pareameters
dataFile = './wse_data.csv';

M = readmatrix(dataFile);

% input dimesion (days)
m_in = 60;
% Try different output dimensions (days)
n_out = 60;

[l_whole, ~] = size(M); %657

% Break the whole dataset in training sessions,
% set training session length
l_sess = 560;%3*m_in + n_out;%50;
% the following test period
l_test = 60;%l_sess;

n_sess = floor((l_whole-l_test)/l_sess); %12


% Normalization flag
norm_fl = 1;
    
% Reformat sequence into input training tensor X of m_in observations dimension, 
% label Y (predicted values) of n_out dimension, on training seesion of
% length l_sess (n_sess of them), and min-max boundaries B for each observation, 
% such that number of observations in a session, including training label sequences
% do not touch test period
[X, Y, B, k_ob] = w_series3_train_tensors(M, m_in, n_out, l_sess, n_sess, norm_fl);

mb_size = 2^floor(log2(k_ob)); %32


% Fit ann into minimal loss function (SSE)
k_hid = m_in;
k_hid1 = m_in + 1;
k_hid2 = 2*m_in + 1;
regNets = cell(n_sess);

max_neuro1 = floor(k_hid1 / n_out);
max_neuro2 = floor(k_hid2 / n_out);

sOptions = trainingOptions('adam', ...
'ExecutionEnvironment','parallel',...
'Shuffle', 'every-epoch',...
'MiniBatchSize', mb_size, ...
'InitialLearnRate',0.02, ...
'MaxEpochs',250);%, ...
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
'MaxEpochs',500);

 sOptionsFinal = trainingOptions('adam', ...
'ExecutionEnvironment','parallel',...
'Shuffle', 'every-epoch',...
'MiniBatchSize', mb_size, ...
'InitialLearnRate',0.01, ...
'MaxEpochs',250);           

%i = 3;
for i = 1:n_sess
    %for k = 1:n_out

    % Start growing GMDH net
    prevLayerName = 'inputFeature';
    oLayers = [
        featureInputLayer(m_in, 'Name', prevLayerName)
    ];
    cgraph = layerGraph(oLayers);


    fprintf('Training net %d\n', i);

    % Start from first GMDH layer
    ll = 1;
    % Target accuracy
    accTarget = 0.3;
    % Stale accuracy chage threshol
    dAccMin = 0.001;
    % Maximal polinomial length (number of chained gmdh layer) 
    lMax = 3;

    [cgraph, regNet, ll, k_hid1_real, curGMDHLayerName, curGMDHRegressionName] =... 
        gmdhLayerGrowN(cgraph, prevLayerName, sOptions, X, Y, m_in, i, n_out, ll, accTarget, max_neuro1, 0, dAccMin, lMax);
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
        accTarget = 0.15; %0.015;% 0.05; 0.08;
        % Stale accuracy chage threshol
        dAccMin = 0.001;
        % Maximal polinomial length (number of chained gmdh layer) 
        lMax = 3;

        [cgraph, regNet, ll, k_hid2_real, curGMDHLayerName, curGMDHRegressionName] =... 
            gmdhLayerGrowN(cgraph, prevLayerName, sOptions2, X, Y, k_hid1_real, i, n_out, ll, accTarget, max_neuro2, 0, dAccMin, lMax);
        prevLayerName = curGMDHLayerName;

        if(k_hid2_real > 1)
            fullyConnectdLastLayerName = 'fullyConnectedLayerLast';
            fullyConnectdLastLayer = fullyConnectedLayer(k_hid2_real, 'Name', fullyConnectdLastLayerName);
            cgraph = addLayers(cgraph, fullyConnectdLastLayer);
            cgraph = connectLayers(cgraph, curGMDHLayerName, fullyConnectdLastLayerName);

            fullyConnectedOutLayerName = 'fcOut';
            fullyConnectedOutLayer = fullyConnectedLayer(n_out,'Name',fullyConnectedOutLayerName);
            cgraph = addLayers(cgraph, fullyConnectedOutLayer);
            cgraph = connectLayers(cgraph, fullyConnectdLastLayerName, fullyConnectedOutLayerName);

            prevLayerName = fullyConnectedOutLayerName;
        end
    end

    regressionLayerName = "regOut";
    regressionLayer = regressionLayer('Name', regressionLayerName);

    cgraph = replaceLayer(cgraph, curGMDHRegressionName, regressionLayer);
    cgraph = connectLayers(cgraph, prevLayerName, regressionLayerName);


    fprintf('Training whole GMDH net %d\n', i);

    regNet = trainNetwork(X(:,:,i)', Y(:,:,i)', cgraph, sOptionsFinal);
    regNets{i} = regNet;

    clear('regressionLayerName');
    clear('regressionLayer');
    clear('cgraph');
    clear('regNet');
    %end
end

%%
[X2, Y2, Yh2, Bt, k_tob] = w_series3_test_tensors(M, m_in, n_out, l_sess, l_test, n_sess, norm_fl);


%% test
%i = 1;
for i = 1:n_sess
    %for k = 1:n_out
        predictedScores = predict(regNets{i}, X2(:, :, i)');
        Y2(:, :, i) = predictedScores';
    %end
end

    
% re-scale in observation bounds
if(norm_fl)
    for i = 1:n_sess
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
w_seriesv_ser_graph(M, Y2, l_whole, l_sess, m_in, n_out, k_tob, n_sess);
