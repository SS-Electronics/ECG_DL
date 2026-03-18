%% Create ECG CNN Layers Function
function layers = create_ecg_cnn_layers(config)
    %CREATE_ECG_CNN_LAYERS Create CNN architecture for ECG classification
    %
    % Input:
    %   config - Configuration structure with architecture parameters
    %
    % Output:
    %   layers - Array of network layers
    %
    % Architecture:
    % Input (1D Conv) -> Conv Blocks -> Global Average Pooling -> FC -> Output
    
    input_length = config.input_length;
    num_leads = config.num_leads;
    num_classes = config.num_classes;
    
    % Get architecture parameters
    filters1 = config.cnn_architecture.num_filters_layer1;
    filters2 = config.cnn_architecture.num_filters_layer2;
    filters3 = config.cnn_architecture.num_filters_layer3;
    filter_size = config.cnn_architecture.filter_size;
    pool_size = config.cnn_architecture.pool_size;
    dropout_rate = config.cnn_architecture.dropout_rate;
    fc_units = config.cnn_architecture.fc_units;
    
    fprintf('\n--- CNN Architecture Configuration ---\n');
    fprintf('Input: [%d leads x %d samples]\n', num_leads, input_length);
    fprintf('Conv Layer 1: %d filters of size %d\n', filters1, filter_size);
    fprintf('Conv Layer 2: %d filters of size %d\n', filters2, filter_size);
    fprintf('Conv Layer 3: %d filters of size %d\n', filters3, filter_size);
    fprintf('Dropout: %.2f\n', dropout_rate);
    fprintf('FC Layer: %d units\n', fc_units);
    fprintf('Output: %d classes\n', num_classes);
    fprintf('--------------------------------------\n\n');
    
    layers = [
        % ===== Input Layer =====
        imageInputLayer([num_leads input_length 1], 'Name', 'input', ...
            'Normalization', 'zscore')
        
        % ===== Convolutional Block 1 =====
        convolution1dLayer(filter_size, filters1, 'Padding', 'same', ...
            'WeightsInitializer', 'he', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling1dLayer(pool_size, 'Stride', pool_size, 'Name', 'pool1')
        dropoutLayer(dropout_rate, 'Name', 'drop1')
        
        % ===== Convolutional Block 2 =====
        convolution1dLayer(filter_size, filters2, 'Padding', 'same', ...
            'WeightsInitializer', 'he', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling1dLayer(pool_size, 'Stride', pool_size, 'Name', 'pool2')
        dropoutLayer(dropout_rate, 'Name', 'drop2')
        
        % ===== Convolutional Block 3 =====
        convolution1dLayer(filter_size, filters3, 'Padding', 'same', ...
            'WeightsInitializer', 'he', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        maxPooling1dLayer(pool_size, 'Stride', pool_size, 'Name', 'pool3')
        dropoutLayer(dropout_rate, 'Name', 'drop3')
        
        % ===== Global Average Pooling =====
        globalAveragePooling1dLayer('Name', 'gap')
        
        % ===== Fully Connected Layers =====
        fullyConnectedLayer(fc_units, 'WeightsInitializer', 'he', 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bnfc1')
        reluLayer('Name', 'relufc1')
        dropoutLayer(dropout_rate, 'Name', 'dropfc1')
        
        % ===== Output Layer =====
        fullyConnectedLayer(num_classes, 'Name', 'fc_output')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classoutput')
    ];
end

%% Create Multi-Scale CNN (ResNet-inspired)
function layers = create_ecg_cnn_resnet(config)
    %CREATE_ECG_CNN_RESNET Create ResNet-inspired CNN for ECG
    %
    % Uses residual connections for improved training
    
    input_length = config.input_length;
    num_leads = config.num_leads;
    num_classes = config.num_classes;
    
    filters1 = config.cnn_architecture.num_filters_layer1;
    filters2 = config.cnn_architecture.num_filters_layer2;
    filters3 = config.cnn_architecture.num_filters_layer3;
    filter_size = config.cnn_architecture.filter_size;
    
    fprintf('\nCreating ResNet-inspired ECG CNN...\n');
    
    layers = [
        % Input
        imageInputLayer([num_leads input_length 1], 'Name', 'input', ...
            'Normalization', 'zscore')
        
        % Initial convolution
        convolution1dLayer(filter_size, filters1, 'Padding', 'same', ...
            'WeightsInitializer', 'he', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        
        % Residual block 1
        convolution1dLayer(filter_size, filters1, 'Padding', 'same', ...
            'WeightsInitializer', 'he', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        additionLayer(2, 'Name', 'add1')
        reluLayer('Name', 'relu2')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        % Residual block 2
        convolution1dLayer(filter_size, filters2, 'Padding', 'same', ...
            'WeightsInitializer', 'he', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        convolution1dLayer(filter_size, filters2, 'Padding', 'same', ...
            'WeightsInitializer', 'he', 'Name', 'conv4')
        batchNormalizationLayer('Name', 'bn4')
        additionLayer(2, 'Name', 'add2')
        reluLayer('Name', 'relu4')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        % Global pooling
        globalAveragePooling1dLayer('Name', 'gap')
        
        % Classification
        fullyConnectedLayer(num_classes, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
end

%% Create Attention-based CNN
function layers = create_ecg_cnn_attention(config)
    %CREATE_ECG_CNN_ATTENTION Create CNN with attention mechanisms
    %
    % Incorporates channel attention for ECG signals
    
    input_length = config.input_length;
    num_leads = config.num_leads;
    num_classes = config.num_classes;
    
    fprintf('\nCreating Attention-based ECG CNN...\n');
    
    filters1 = config.cnn_architecture.num_filters_layer1;
    filters2 = config.cnn_architecture.num_filters_layer2;
    filter_size = config.cnn_architecture.filter_size;
    
    layers = [
        imageInputLayer([num_leads input_length 1], 'Name', 'input', ...
            'Normalization', 'zscore')
        
        % Conv block 1
        convolution1dLayer(filter_size, filters1, 'Padding', 'same', ...
            'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        % Conv block 2
        convolution1dLayer(filter_size, filters2, 'Padding', 'same', ...
            'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        % Global pooling
        globalAveragePooling1dLayer('Name', 'gap')
        
        % Dense layers
        fullyConnectedLayer(128, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bnfc1')
        reluLayer('Name', 'relufc1')
        dropoutLayer(0.3, 'Name', 'dropfc1')
        
        % Output
        fullyConnectedLayer(num_classes, 'Name', 'fc_output')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classoutput')
    ];
end
