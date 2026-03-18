classdef CNN1D < BaseModel
    % CNN1D - 1D Convolutional Neural Network for ECG classification
    % Inherits from BaseModel
    
    properties
        conv_layers         % Convolutional layers
        pool_layers         % Pooling layers
        dense_layers        % Fully connected layers
        weights             % Network weights
        biases              % Network biases
        activations         % Activation function handles
    end
    
    methods
        function obj = CNN1D(model_params)
            % Initialize 1D CNN model
            obj = obj@BaseModel('CNN1D', model_params);
            obj.createArchitecture();
        end
        
        function createArchitecture(obj)
            % Create 1D CNN architecture for ECG
            % Input: (10000 samples x 12 leads)
            % Output: (num_classes)
            
            params = obj.model_params;
            
            % Layer 1: Conv + Pool
            % Conv: 32 filters, kernel=50, stride=2
            obj.conv_layers(1).filters = 32;
            obj.conv_layers(1).kernel_size = 50;
            obj.conv_layers(1).stride = 2;
            obj.conv_layers(1).padding = 'same';
            obj.conv_layers(1).activation = 'relu';
            
            obj.pool_layers(1).type = 'maxpool';
            obj.pool_layers(1).kernel_size = 4;
            obj.pool_layers(1).stride = 4;
            
            % Layer 2: Conv + Pool
            % Conv: 64 filters, kernel=30, stride=1
            obj.conv_layers(2).filters = 64;
            obj.conv_layers(2).kernel_size = 30;
            obj.conv_layers(2).stride = 1;
            obj.conv_layers(2).padding = 'same';
            obj.conv_layers(2).activation = 'relu';
            
            obj.pool_layers(2).type = 'maxpool';
            obj.pool_layers(2).kernel_size = 4;
            obj.pool_layers(2).stride = 4;
            
            % Layer 3: Conv + Pool
            % Conv: 128 filters, kernel=20, stride=1
            obj.conv_layers(3).filters = 128;
            obj.conv_layers(3).kernel_size = 20;
            obj.conv_layers(3).stride = 1;
            obj.conv_layers(3).padding = 'same';
            obj.conv_layers(3).activation = 'relu';
            
            obj.pool_layers(3).type = 'maxpool';
            obj.pool_layers(3).kernel_size = 4;
            obj.pool_layers(3).stride = 4;
            
            % Flatten + Dense layers
            % Dense: 256 units
            obj.dense_layers(1).units = 256;
            obj.dense_layers(1).activation = 'relu';
            obj.dense_layers(1).dropout = params.dropout_rate;
            
            % Dense: 128 units
            obj.dense_layers(2).units = 128;
            obj.dense_layers(2).activation = 'relu';
            obj.dense_layers(2).dropout = params.dropout_rate;
            
            % Output layer
            obj.dense_layers(3).units = params.num_classes;
            obj.dense_layers(3).activation = 'softmax';
            obj.dense_layers(3).dropout = 0;
            
            fprintf('[CNN1D] Architecture created\n');
            fprintf('  Conv layers: %d\n', length(obj.conv_layers));
            fprintf('  Dense layers: %d\n', length(obj.dense_layers));
            fprintf('  Output classes: %d\n', params.num_classes);
        end
        
        function output = predict(obj, input_data)
            % Forward pass through the network
            % Input: (samples x leads x batch_size) or (samples x leads)
            
            % Handle single sample vs batch
            if ndims(input_data) == 2
                input_data = cat(3, input_data);
            end
            
            batch_size = size(input_data, 3);
            
            % Reshape for CNN: (samples x leads x batch_size) -> (samples x 1 x leads x batch_size)
            x = permute(input_data, [1, 3, 2]);  % (samples x batch_size x leads)
            x = reshape(x, [size(x,1), 1, size(x,3), batch_size]);  % (samples x 1 x leads x batch_size)
            
            % Convolutional blocks
            for i = 1:length(obj.conv_layers)
                % Convolution
                x = obj.conv1d(x, obj.conv_layers(i));
                
                % Activation
                x = obj.applyActivation(x, obj.conv_layers(i).activation);
                
                % Pooling
                x = obj.maxpool1d(x, obj.pool_layers(i));
            end
            
            % Flatten
            x = reshape(x, [size(x,1) * size(x,2) * size(x,3), batch_size]);
            
            % Dense layers
            for i = 1:length(obj.dense_layers)
                % Apply dropout during training if not trained
                if ~obj.is_trained && i < length(obj.dense_layers)
                    dropout_mask = rand(size(x,1), 1) > obj.dense_layers(i).dropout;
                    dropout_mask = dropout_mask / (1 - obj.dense_layers(i).dropout);
                    x = x .* dropout_mask;
                end
                
                % Linear transformation (simplified - using identity for demo)
                x = x + randn(size(x)) * 0.01;  % Simplified forward pass
                
                % Activation
                x = obj.applyActivation(x, obj.dense_layers(i).activation);
            end
            
            output = x';  % (batch_size x num_classes)
        end
        
        function output = conv1d(obj, input, conv_config)
            % 1D Convolution operation
            % Input: (seq_length x 1 x channels x batch_size)
            
            [seq_len, ~, in_channels, batch_size] = size(input);
            kernel_size = conv_config.kernel_size;
            stride = conv_config.stride;
            n_filters = conv_config.filters;
            
            % Simplified convolution (using correlation)
            % In practice, use MATLAB's built-in convolution or deep learning tools
            output_len = floor((seq_len - kernel_size) / stride) + 1;
            output = randn(output_len, 1, n_filters, batch_size) * 0.1;  % Placeholder
        end
        
        function output = maxpool1d(obj, input, pool_config)
            % 1D Max pooling
            % Input: (seq_length x 1 x channels x batch_size)
            
            [seq_len, ~, channels, batch_size] = size(input);
            kernel_size = pool_config.kernel_size;
            stride = pool_config.stride;
            
            output_len = floor((seq_len - kernel_size) / stride) + 1;
            output = zeros(output_len, 1, channels, batch_size);
            
            for t = 1:output_len
                start_idx = (t - 1) * stride + 1;
                end_idx = start_idx + kernel_size - 1;
                window = input(start_idx:end_idx, 1, :, :);
                output(t, 1, :, :) = max(window, [], 1);
            end
        end
        
        function output = applyActivation(obj, input, activation)
            % Apply activation function
            switch lower(activation)
                case 'relu'
                    output = max(input, 0);
                case 'sigmoid'
                    output = 1 ./ (1 + exp(-input));
                case 'tanh'
                    output = tanh(input);
                case 'softmax'
                    exp_input = exp(input - max(input, [], 1));
                    output = exp_input ./ sum(exp_input, 1);
                otherwise
                    output = input;
            end
        end
        
        function visualizeArchitecture(obj)
            % Display network architecture
            fprintf('\n========== CNN1D Architecture ==========\n');
            fprintf('Input: (%d samples x %d leads)\n', ...
                obj.model_params.input_size, obj.model_params.num_leads);
            
            fprintf('\nConvolutional Blocks:\n');
            for i = 1:length(obj.conv_layers)
                fprintf('  Conv%d: %d filters, kernel=%d, stride=%d, activation=%s\n', ...
                    i, obj.conv_layers(i).filters, obj.conv_layers(i).kernel_size, ...
                    obj.conv_layers(i).stride, obj.conv_layers(i).activation);
                fprintf('  Pool%d: maxpool, kernel=%d, stride=%d\n', ...
                    i, obj.pool_layers(i).kernel_size, obj.pool_layers(i).stride);
            end
            
            fprintf('\nDense Layers:\n');
            for i = 1:length(obj.dense_layers)
                fprintf('  Dense%d: %d units, activation=%s, dropout=%.2f\n', ...
                    i, obj.dense_layers(i).units, obj.dense_layers(i).activation, ...
                    obj.dense_layers(i).dropout);
            end
            
            fprintf('Output: %d classes\n', obj.model_params.num_classes);
            fprintf('========================================\n\n');
        end
    end
end
