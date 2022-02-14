function [cgraph, regNet, ll, nOut, curGMDHLayerName, curGMDHRegressionName] =...
    gmdhLayerGrowN(cgraph, prevLayerName, sOptions, X, Y, m_in, i_sess, n_out, ll, accTarget, nTarget, nMin, dAccMin, lMax)

    % Starting placeholder for accuracy
    %accCurr = 1.;
    %num_out_old = nTarget;
    accNext = accTarget;
    i = i_sess;
    %k = k_out;

    % Add first GMDH layer later in the loop, we may more of them in
    % the loop
    curGMDHLayerNamePattern = 'gmdhLayer';

    curGMDHLayerName = sprintf('%s%d', curGMDHLayerNamePattern, ll);
    
    curGMDHLayer = gmdhLayerN(curGMDHLayerName, m_in, n_out);
    cgraph = addLayers(cgraph, curGMDHLayer);

    num_out_old = sum(cgraph.Layers(strcmp({cgraph.Layers.Name}, curGMDHLayerName)).numOutChannels, 1);
    
    % Add GMDH Regression layer to the graph here, later we will
    % connect/disconnect it with GMDH layer in the middle
    curGMDHRegressionName = 'gmdhRegression';
    if(ll == 1)
        curGMDHRegression = gmdhRegression(curGMDHRegressionName);
        cgraph = addLayers(cgraph, curGMDHRegression);
    end

    for l = 1:lMax

        % Add first/additional GMDH leyer(s) and connect it with final GMDH
        % Regression layer
        %cgraph = addLayers(cgraph, curGMDHLayer);
        cgraph = connectLayers(cgraph, prevLayerName, curGMDHLayerName);
        cgraph = connectLayers(cgraph, curGMDHLayerName, curGMDHRegressionName);

        % Multiply training output regression values - each output is just a
        % candidate for the best approximation of the same training value
        %num_out = cgraph.Layers(strcmp({cgraph.Layers.Name}, curGMDHLayerName)).numOutChannels;
        %num_out_old = num_out;
        %fprintf('Training %d layer %d neurons, GMDH net %d\n', l, num_out, i);
        %Ytr = repmat(Y(k,:,i), [num_out, 1]);

        [Ytr, num_out] = cgraph.Layers(strcmp({cgraph.Layers.Name}, curGMDHLayerName)).buildYtr(Y(:,:,i));
        fprintf('Training level %d, %d->%d neurons, session %d\n', l, num_out_old, num_out, i);
        num_out_old = num_out;
        regNet = trainNetwork(X(:,:,i)', Ytr', cgraph, sOptions);

        % Find errors for each output channel and prune worse polynomial
        % candidate paths in the last GMDH layer
        [m, n] = size (Ytr);
        MSE = sum((activations(regNet, X(:,:,i)', curGMDHLayerName) - Ytr).^2, 2)/n;
        %accOld = accCurr;
        accCurr = sum(MSE)/m;

        fprintf('Accuracy %f -> %f, level %d, %d neurons, session %d\n', accCurr, accNext, l, num_out, i);

        modLayer = regNet.Layers( strcmp({regNet.Layers.Name}, curGMDHLayerName) );
        if(nTarget)
            if(accTarget)
                modLayer = modLayer.pruneNACC(nTarget, nMin, accNext, MSE);
            else
                modLayer = modLayer.pruneN(int16(nTarget), MSE);
            end
        else
            modLayer = modLayer.pruneNACC(int16(modLayer.numOutChannels(1)/2), nMin, accNext, MSE);
        end

        cgraph = replaceLayer(cgraph, curGMDHLayerName, modLayer);

        %figure
        %plot(cgraph);

        % Retrain the pruned Net again
        %num_outp = cgraph.Layers(strcmp({cgraph.Layers.Name}, curGMDHLayerName)).numOutChannels;
        %fprintf('Training %d pruned layer %d neurons, GMDH net %d %d\n', l, num_outp, i, k);
        %Ytr = repmat(Y(k,:,i), [num_outp, 1]);

        [Ytr, num_outp] = cgraph.Layers(strcmp({cgraph.Layers.Name}, curGMDHLayerName)).buildYtr(Y(:,:,i));
        fprintf('Training level %d, %d->%d neurons, session %d\n', l, num_out, num_outp, i);
        regNet = trainNetwork(X(:,:,i)', Ytr', cgraph, sOptions);

        [m, n] = size (Ytr);
        MSE = sum((activations(regNet, X(:,:,i)', curGMDHLayerName) - Ytr).^2, 2)/n;
        accCurr = sum(MSE)/m;

        fprintf('Accuracy %f -> %f, level %d, pruned %d neurons, session %d\n', accCurr, accNext, l, num_outp, i);
        % Bail out if only one pair left, training accuracy falls or stagnant, 
        % or reached desired level
        cgraph = disconnectLayers(cgraph, curGMDHLayerName, curGMDHRegressionName);
        nOut = sum(modLayer.numOutChannels, 1);

        
        % Exit on accuracy degradation
        if (l > 1) && ( (saveAccCurr < accCurr) || ((saveAccCurr - accCurr) < dAccMin) )
            cgraph = saveCgraph;
            regNet = saveRegNet; 
            ll = saveLl;
            nOut = saveNOut; 
            curGMDHLayerName = saveCurGMDHLayerName;
            curGMDHRegressionName = saveCurGMDHRegressionName;
            break;
        end

        if (l == lMax) || (sum(modLayer.numOutChannels, 1) == modLayer.numOutProd) || (accCurr < accNext) || (num_out_old == num_outp) 
            break;
        end

        saveCgraph = cgraph;
        saveRegNet = regNet; 
        saveLl = ll;
        saveNOut = nOut; 
        saveCurGMDHLayerName = curGMDHLayerName;
        saveCurGMDHRegressionName = curGMDHRegressionName;
        saveAccCurr = accCurr;


        prevLayerName = sprintf('%s%d', curGMDHLayerNamePattern, ll);
        ll = ll + 1;
        curGMDHLayerName = sprintf('%s%d', curGMDHLayerNamePattern, ll);
        curGMDHLayer = gmdhLayerN(curGMDHLayerName, sum(modLayer.numOutChannels, 1), n_out);

        cgraph = addLayers(cgraph, curGMDHLayer);

        accNext = accNext/2;
    end
end