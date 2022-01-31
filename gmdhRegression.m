classdef gmdhRegression < nnet.layer.RegressionLayer
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.

        % Number output and input channels
        %numOutChannels 
        %numInChannels

        %errSquaresPair
    end
 
    methods
        function layer = gmdhRegression(name)           
            % (Optional) Create a myRegressionLayer.

            % Layer constructor function goes here.

            % Set layer name.
            layer.Name = name;

            %layer.numInChannels = numInChannels;
            %layer.numOutChannels = numInChannels * (numInChannels - 1) / 2;
            
            %layer.errSquaresPair = zeros([layer.numOutChannels, 1]);
        end

        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the training
            % targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here.
            
            % calculate channel-wise (row) errors
            [m, r, ~] = size(Y);
            %fprintf('gmdhReg fwd loss m=%d r=%d n=%d\n', m, r, n);  n=1

            errSquaresPair = sum((Y-T) .^ 2, 1) / m;

            %summarise parallel observations (minibatches)
            % Take mean over mini-batch
            loss = sum(errSquaresPair, 2) / r;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % (Optional) Backward propagate the derivative of the loss 
            % function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the 
            %                 predictions Y        

            % Layer backward loss function goes here.

            % calculate channel-wise (row) error dervivatives
            %Y
            [m, r, ~] = size(Y);
            %fprintf('gmdhReg back loss m=%d r=%d n=%d\n', m, r, n);  n=1

            errSquareDer = 2. * (Y-T);
            dLdY = errSquareDer;

        end

    end
end