classdef vRegression < nnet.layer.RegressionLayer
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.

        % Number output and input channels
        numInChannels 
        numInVelChannels

        %errSquaresPair
    end
 
    methods
        function layer = vRegression(name, numInVelChannels)           
            % (Optional) Create a myRegressionLayer.

            % Layer constructor function goes here.

            % Set layer name.
            layer.Name = name;

            %layer.numInChannels = numInChannels;
            layer.numInVelChannels = numInVelChannels;
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
            
            % dimensions: r - channels(row) errors
            [m, r, ~] = size(Y);
            %fprintf('vReg fwd loss m=%d r=%d n=%d\n', m, r, n);  n=1


            %number of observation (noit including velocities)
            mOb = m - double(layer.numInVelChannels);

            % SSE on location observatrion
            ObErr = Y(1:mOb) - T(1:mOb);
            ObMSE = sum(ObErr .^ 2, 1) / mOb;

            % SSE on ANN prediction and expected kinematic location
            mVel = double(layer.numInVelChannels);
            Yk = Y(1:mOb-1) + Y(mOb+1:mOb+mVel);
            KinErr = Y(2:mOb) - Yk;
            KinMSE = sum(KinErr .^2, 1) / mVel;

            % SSE on velocity observation
            VelErr = Y(mOb+1:mOb+mVel) - T(mOb+1:mOb+mVel);
            VelMSE = sum(VelErr .^ 2, 1) / mVel;


            %summarise parallel observations (minibatches)
            % Take mean over mini-batch
            loss = (sum(ObMSE, 2) + sum(KinMSE, 2) + sum(VelMSE, 2))/ r;
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

            %number of observation (noit including velocities)
            mOb = m - double(layer.numInVelChannels);

            % dSSE on observatrion - same as beginning of errSquareDer
            %ObErr = Y(1:mOb,:) - T(1:mOb,:);
            %dObErr = 2. * ObErr;

            % dSSE on ANN prediction and kinematic vales
            mVel = double(layer.numInVelChannels);
            %fprintf('vReg back loss m=%d r=%d mOb=%d mVel=%d\n', m, r, mOb, mVel); 

            Yk = Y(1:mOb-1) + Y(mOb+1:mOb+mVel);
            KinErr = Y(2:mOb) - Yk;
            dKinErr = 2. * KinErr;

            errSquareDer = 2. * (Y-T);
            errSquareDer(2:mOb) = errSquareDer(2:mOb) + dKinErr;

            dLdY = errSquareDer;

        end

    end
end