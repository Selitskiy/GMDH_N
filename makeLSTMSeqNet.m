function regNet = makeLSTMSeqNet(i, m_in, n_out, k_hid1, k_hid2, mb_size, X, Ys)

    sLayers = [
        sequenceInputLayer(1)
            %gruLayer(k_hid1)%, 'OutputMode','last')
            %gruLayer(k_hid2)%, 'OutputMode','last')
            lstmLayer(k_hid1)%, 'OutputMode','last')
            lstmLayer(k_hid2)%, 'OutputMode','last')
            fullyConnectedLayer(1)
            regressionLayer
    ];

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','auto',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_size, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',1000);

    fprintf('Training LSTM Seq net %d\n', i);

    CX = num2cell(X(:, :, i)', 2);
    CY = num2cell(Ys(:, :, i)', 2);

    regNet = trainNetwork(CX, CY, sLayers, sOptions);
    
end