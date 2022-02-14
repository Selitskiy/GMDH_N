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
%saveDataPrefix = 'nikkey4030_';
saveDataPrefix = 'dax4030_';

%saveDataPrefix = '7203toyota_';
%saveDataPrefix = 'nvidia_';
%saveDataPrefix = 'tsla4030_';

%saveDataPrefix = 'AirPassengers1_114_30_';
%saveDataPrefix = 'sun_1_';
%saveDataPrefix = 'SN_y_tot_V2.0_spots_4030_';

save_regNet_fileT = '~/data/ws_';

%dataFile = 'nasdaq_1_3_05-1_28_22.csv';%'./wse_data.csv';
%dataFile = 'dj_1_3_05-1_28_22.csv';
%dataFile = 'nikkei_1_4_05_1_31_22.csv';
dataFile = 'dax_1_3_05_1_31_22.csv';

%dataFile = '7203toyota_1_4_05_1_31_22';
%dataFile = 'nvidia_1_3_05_1_28_22';
%dataFile = 'tsla_6_30_10_1_28_22.csv';

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
[X, Ys, Y, B, k_ob] = w_series3_train_tensors(M, m_in, n_out, l_sess, n_sess, norm_fl);


% Fit ann into minimal loss function (SSE)
mult = 1;
%k_hid = floor(mult * m_ine);
k_hid1 = floor(mult * (m_in + 1));
k_hid2 = floor(mult * (2*m_in + 1));
regNets = cell(n_sess);


%% regNet parameters
modelName = 'lstm_seq';
%modelName = 'gmdh';

mb_size = 2^floor(log2(k_ob)); %32
       

%% Train or pre-load regNets
for i = 1:n_sess

    save_regNet_file = strcat(save_regNet_fileT, modelName, '_', saveDataPrefix, int2str(i), '_', int2str(m_in), '_', int2str(n_out), '_', int2str(n_sess), '.mat');
    %save_regNet_file = strcat(save_regNet_fileT, modelName, '_', saveDataPrefix, int2str(i), '_', int2str(m_in), '_', int2str(n_out), '_71', '.mat');
    if isfile(save_regNet_file)
        fprintf('Loading net %d from %s\n', i, save_regNet_file);
        load(save_regNet_file, 'regNet');
    else
        clear('regNet');
    end


    if exist('regNet') == 0

        %regNet = makeANNNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        %regNet = makeReLUNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        %regNet = makeTanhNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        regNet = makeLSTMSeqNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);
        %regNet = makeGMDHNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y);

        save(save_regNet_file, 'regNet');
    end

    regNets{i} = regNet;

    %clear('cgraph');
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
% Number of training sessions with following full-size test sessions 
t_sess = floor((l_whole - m_in - n_out) / l_sess);

[X2, Y2s, Y2, Yh2, Bt, k_tob] = w_series4_test_tensors(M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl, 0);


%% test
%for i = 1:t_sess-sess_off
%        predictedScores = predict(regNets{i}, X2(:, :, i)');
%        Y2(:, :, i) = predictedScores';
%end

    % LSTM Seq test
    for i = 1:t_sess-sess_off
        
        % Now feeding test data
        for j = 1:k_tob
            lstmNet = resetState(regNets{i});
            % Trick - go through all input sequence to get the first next
            % prediction outside the seqence, discarding intermediate
            % predictions
            %or k = 1:m_in
            %    [lstmNet, Y2(1, j, i)] = predictAndUpdateState(lstmNet, X2(k, j, i));
            %end
            [lstmNet, Y2s(:, j, i)] = predictAndUpdateState(lstmNet, X2(:, j, i)');
            Y2(1, j, i) = Y2s(end, j, i);

            % Continue predicting further output points based on previous
            % predicvtion
            for l = 2:n_out
                %Y2s(1:m_in-l+1, j, i) = X2(l:end, j, i);
                [lstmNet, Y2s(:, j, i)] = predictAndUpdateState(lstmNet, Y2s(:, j, i)');
                Y2(l, j, i) = Y2s(end, j, i);
            end

        end
    end


%% re-scale in observation bounds
if(norm_fl)
    for i = 1:t_sess-sess_off
        for j = 1:k_tob
            Y2(:, j, i) = w_series2_rescale(Y2(:, j, i), Bt(1,j,i), Bt(2,j,i));
            Yh2(:, j, i) = w_series2_rescale(Yh2(:, j, i), Bt(1,j,i), Bt(2,j,i));
        end
    end
end


%% Calculate errors
[S2, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = w_seriesv_calc_err(Y2, Yh2, n_out); 

fprintf('%s, dataFN %s, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, Err: %f\n', modelName, dataFile, m_in, n_out, n_sess, t_sess, S2);

%% Error and Series Plot
%w_series2_err_graph(Y2, Yh2);
w_seriesva_ser_graph(M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg);

