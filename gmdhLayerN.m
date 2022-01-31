classdef gmdhLayerN < nnet.layer.Layer

    properties
        % (Optional) Layer properties.

        % Layer properties go here.

        % Pairwise weight mask and its transpose
        M
        %MT
        
        % Number output and input channels
        numOutChannels 
        numInChannels

        % Product size - target output dimension
        numOutProd
        
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
        
        % Weights
        W
        %Bias
        W0
    end
    
    methods
        function layer = gmdhLayerN(name, numInChannels, numOutProd)
            % (Optional) Create a myLayer.
            % This function must have the same name as the layer.

            % Layer constructor function goes here.
            
            % layer =  PairwiseLayer(numChannels) creates a Pairwise 
            % hidden layer with numInChannels (and driven by it numOutChannels) 
            % channels, and layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Pairwise 2nd degree polinomial GMDH: chennel-wise SSE, n=" + numInChannels;
        
            layer.numOutProd = numOutProd;

            layer.numInChannels = repmat( numInChannels, [numOutProd, 1] );
            layer.numOutChannels = repmat( numInChannels * (numInChannels - 1) / 2, [numOutProd, 1] );

            
            
            
            %Initialize weight mask
            M = zeros([layer.numOutChannels(1), numInChannels]);
            
            % k iterates from 1 to numMidChannels
            k = 0;
            
            % i - first input in the pair
            for i = 1 : numInChannels-1
                
                % j - second input in the pair
                for j = i+1 : numInChannels
                    k = k + 1;
                    
                    % mask for first inpu
                    M(k, i) = 1;
                    
                    %mask for second input
                    M(k, j) = 1;
                   
                end
            end


            layer.M = repmat( M, [numOutProd, 1] );
         
            
            % Initialize weight coefficients.
            layer.W = layer.M .* rand([layer.numOutChannels(1) * numOutProd, numInChannels]);

            layer.W0 = rand([layer.numOutChannels(1) * numOutProd, 1]);
            
        end


        function [Ytr, num_out] = buildYtr(layer, Y)
            num_out = sum(layer.numOutChannels, 1);
            [~, m] = size(Y);
            Ytr = zeros([num_out, m]);

            offset = 0;
            for i = 1:layer.numOutProd
                n_cur = layer.numOutChannels(i);

                Ytr(offset+1:offset+n_cur,:) = repmat(Y(i,:), [n_cur, 1]);

                offset = offset + layer.numOutChannels(i);
            end

        end


        function layer = pruneN(layer, n1, MSE)
            n = repmat(n1, [layer.numOutProd, 1]);
            IGlob = zeros([sum(n,1), 1]);

            offset = 0;
            for i = 1:layer.numOutProd
                n_cur = layer.numOutChannels(i);
                if(n_cur < n(i))
                    n(i) = n_cur;
                end

                [aMSE, I] = sort(MSE(offset+1:offset+n_cur), 1, 'ascend');
                
                newI = I(1:n(i));

                m = size(newI);

                IGlob(offset+1:offset+m) = newI; 

                offset = offset + layer.numOutChannels(i);
                layer.numOutChannels(i) = n(i);
            end

            % compact unused indexes
            IGlob = IGlob( IGlob > 0 );

            layer.M = layer.M( IGlob, :);
            layer.W = layer.W( IGlob, :);
            layer.W0 = layer.W0( IGlob);
        end

        function layer = pruneNACC(layer, n1, n_min1, acc, MSE)
            n = repmat(n1, [layer.numOutProd, 1]);
            n_min = repmat(n_min1, [layer.numOutProd, 1]);
            IGlob = zeros([sum(n,1), 1]);

            offset = 0;
            for i = 1:layer.numOutProd
                n_cur = layer.numOutChannels(i);
                if(n_cur < n(i))
                    n(i) = n_cur;
                end
    
                if(n_min(i) > n(i))
                    n_min(i) = n(i);
                end

                [aMSE, I] = sort(MSE(offset+1:offset+n_cur), 1, 'ascend');
                newACCI = I(aMSE < acc);
                [ACCn, ~] = size(newACCI);

                if (n_min(i) > 0) && (ACCn < n_min(i))
                    newI = I(1:n_min(i));
                    n(i) = n_min(i);
                elseif ACCn < 1
                    newI = I(1:1);
                    n(i) = 1;
                elseif ACCn < n(i)
                    newI = newACCI;
                    n(i) = ACCn;
                else
                    newI = I(1:n(i));
                end

                m = size(newI);

                IGlob(offset+1:offset+m) = newI; 
                
                offset = offset + layer.numOutChannels(i);
                layer.numOutChannels(i) = n(i);
            end

            % compact unused indexes
            IGlob = IGlob( IGlob > 0 );

            layer.M = layer.M( IGlob, :);
            layer.W = layer.W( IGlob, :);
            layer.W0 = layer.W0( IGlob );
        end


        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - (Optional) Updated layer state.
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer predict function here.

            [c, n] = size(X);
            
            %if class(X) == "gpuArray"
            %    YR = (layer.W .* gpuArray(layer.M)) * (X .* X) + layer.W0;
            %else
            %    YR = (layer.W .* layer.M) * (X .* X) + layer.W0;
            %end   

            if class(X) == "gpuArray"
                Y = (layer.W .* gpuArray(layer.M)) * X + layer.W0;
            else
                Y = (layer.W .* layer.M) * X + layer.W0;
            end
            %YR = Y .* Y + layer.W0;

            Z = Y .* Y;% + layer.W0; %YR;          
            
        end

%function [dLdX, dLdW] =... 
%        backward(layer, X, Z, dLdZ, memory)
            
%    [h, w, c, n] = size(X);
            
%    dLdZR = reshape(dLdZ,[],n); %squeeze(dLdZ); 
            
%    layer.W = layer.M .* layer.W;
%    layer.NW = layer.N * layer.W;
            
    %chain dLdY = J(dz/dy)T * dLdZ
%    if class(X) == "gpuArray"
%        dLdX = reshape(gpuArray(layer.NW') * dLdZR,...
%            [h, w, c, n]);
%    else
%        dLdX = reshape(layer.NW' * dLdZR,...
%            [h, w, c, n]);
%    end             
                                     
    %chain dLdW = J(dz/dW)T * dLdZ
%    [cw, nw] = size(layer.W);
%    XR = squeeze(X);
%    if class(X) == "gpuArray"
%        dLdW = gpuArray(layer.W) .* (sum(XR,2)/n .*... 
%            ones([cw, nw], 'like', X)')' .*... 
%            (layer.N' * sum(dLdZR,2)/n);
%    else
%        dLdW = layer.W .* (sum(XR,2)/n .*...
%            ones([cw, nw], 'like', X)')' .*...
%            (layer.N' * sum(dLdZR,2)/n);
%    end            
            
%end

    end
end