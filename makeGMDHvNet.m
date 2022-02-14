function [regNet, cgraph] = makeGMDHvNet(i, m_ine, n_oute, n_out, k_hid1, k_hid2, mb_size, X, Y)
    
sOptions = trainingOptions('adam', ...
'ExecutionEnvironment','parallel',...
'Shuffle', 'every-epoch',...
'MiniBatchSize', mb_size, ...
'InitialLearnRate',0.02, ...
'MaxEpochs',500);

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

    max_neuro1 = floor(k_hid1 / n_out);
    max_neuro2 = floor(k_hid2 / n_out);
    min_neuro1 = 0;
    min_neuro2 = 0;

    % Number of velocity outputs
    n_outv = n_oute - n_out;


    [cgraph, regNet, ll, k_hid1_real, curGMDHLayerName, curGMDHRegressionName] =... 
        gmdhLayerGrowN(cgraph, prevLayerName, sOptions, X, Y, m_ine, i, n_oute, ll, accTarget, max_neuro1, min_neuro1, dAccMin, lMax);
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
            gmdhLayerGrowN(cgraph, prevLayerName, sOptions2, X, Y, k_hid1_real, i, n_oute, ll, accTarget, max_neuro2, min_neuro2, dAccMin, lMax);
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
end