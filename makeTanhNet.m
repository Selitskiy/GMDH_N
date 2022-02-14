function regNet = makeTanhNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y)

    sLayers = [
        featureInputLayer(m_in)
        fullyConnectedLayer(k_hid1)
        tanhLayer
        fullyConnectedLayer(k_hid2)
        tanhLayer
        fullyConnectedLayer(n_out)
        regressionLayer
    ];

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','parallel',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',1000);

    fprintf('Training ANN Reg net %d\n', i);   

    regNet = trainNetwork(X(:, :, i)', Y(:, :, i)', sLayers, sOptions);
    
end