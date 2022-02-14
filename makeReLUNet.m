function regNet = makeReLUNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Y)

    sLayers = [
                    sequenceInputLayer(1)
            %gruLayer(k_hid1)%, 'OutputMode','last')
            %gruLayer(k_hid2)%, 'OutputMode','last')
            lstmLayer(k_hid1)%, 'OutputMode','last')
            lstmLayer(k_hid2)%, 'OutputMode','last')
            %fullyConnectedLayer(10)
            %reluLayer
            fullyConnectedLayer(1)
            regressionLayer

        featureInputLayer(m_in)
        fullyConnectedLayer(k_hid1)
        reluLayer
        fullyConnectedLayer(k_hid2)
        reluLayer
        fullyConnectedLayer(n_out)
        regressionLayer
    ];

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','parallel',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',1000);

    fprintf('Training ReLU net %d\n', i);   

    regNet = trainNetwork(X(:, :, i)', Y(:, :, i)', sLayers, sOptions);
    
end