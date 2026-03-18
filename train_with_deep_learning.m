%% ECG CNN CLASSIFICATION - COMPLETE TRAINING WITH DEEP LEARNING TOOLBOX
% Full implementation with actual neural network training
% Uses Deep Learning Toolbox for real CNN training

clear; clc; close all;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  ECG CNN Classification - Full Training Pipeline              ║\n');
fprintf('║  WITH Deep Learning Toolbox                                   ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% STEP 1: Initialize Project
fprintf('STEP 1: PROJECT INITIALIZATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

ProjectConfig.initialize();
fprintf('✓ Configuration loaded\n\n');

%% STEP 2: Generate Training Data
fprintf('STEP 2: DATA PREPARATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('2.1 Generating synthetic ECG training data...\n');

signal_length = 10000;
num_leads = 12;
n_samples = 30;  % 30 patients (5 per disease class)

% Generate data
t = (0:signal_length-1)' / 1000;
all_data = zeros(signal_length, num_leads, n_samples);
all_labels = zeros(n_samples, 6);

for sample = 1:n_samples
    disease_class = mod(sample-1, 6) + 1;
    signal = randn(signal_length, num_leads) * 0.1;
    
    for lead = 1:num_leads
        if disease_class == 1      % Normal
            ecg = 0.8*sin(2*pi*1*t) + 0.3*sin(4*pi*1*t);
        elseif disease_class == 2  % MI
            ecg = 0.5*sin(2*pi*1*t) - 0.3;
        elseif disease_class == 3  % LBBB
            ecg = 1.2*sin(2*pi*1*t);
        elseif disease_class == 4  % RBBB
            ecg = 0.9*sin(2*pi*1*t) + 0.2*sin(6*pi*1*t);
        elseif disease_class == 5  % SB
            ecg = 0.7*sin(2*pi*0.67*t);
        else                        % AF
            irregular_freq = 1 + 0.3*randn();
            ecg = 0.5*sin(2*pi*irregular_freq*t) + 0.2*randn(signal_length, 1);
        end
        
        signal(:, lead) = signal(:, lead) + ecg;
        signal(:, lead) = (signal(:, lead) - mean(signal(:, lead))) / (std(signal(:, lead)) + 1e-6);
    end
    
    all_data(:, :, sample) = signal;
    all_labels(sample, disease_class) = 1;
end

fprintf('✓ Generated %d synthetic ECG samples\n', n_samples);
fprintf('  Shape: (%d, %d, %d)\n\n', size(all_data, 1), size(all_data, 2), size(all_data, 3));

%% STEP 3: Data Augmentation
fprintf('STEP 3: DATA AUGMENTATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('3.1 Augmenting data (2x)...\n');

augmented_data = all_data;
for sample = 1:n_samples
    signal = all_data(:, :, sample);
    signal = signal * (0.95 + rand() * 0.1);
    signal = signal + randn(size(signal)) * 0.01;
    augmented_data = cat(3, augmented_data, signal);
end

augmented_labels = repmat(all_labels, 2, 1);

fprintf('✓ Augmentation complete: %d → %d samples\n\n', n_samples, size(augmented_data, 3));

%% STEP 4: Data Split
fprintf('STEP 4: DATA SPLITTING\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('4.1 Creating train/val/test split...\n');

n_total = size(augmented_data, 3);
n_train = floor(n_total * 0.6);
n_val = floor(n_total * 0.2);
n_test = n_total - n_train - n_val;

idx = randperm(n_total);
idx_train = idx(1:n_train);
idx_val = idx(n_train+1:n_train+n_val);
idx_test = idx(n_train+n_val+1:end);

train_data = augmented_data(:, :, idx_train);
val_data = augmented_data(:, :, idx_val);
test_data = augmented_data(:, :, idx_test);

train_labels = augmented_labels(idx_train, :);
val_labels = augmented_labels(idx_val, :);
test_labels = augmented_labels(idx_test, :);

fprintf('✓ Split complete\n');
fprintf('  Train: %d | Val: %d | Test: %d\n\n', ...
    size(train_data, 3), size(val_data, 3), size(test_data, 3));

%% STEP 5: Convert Labels to Categorical
fprintf('STEP 5: CONVERT LABELS\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('5.1 Converting one-hot to categorical...\n');

[~, train_class] = max(train_labels, [], 2);
[~, val_class] = max(val_labels, [], 2);
[~, test_class] = max(test_labels, [], 2);

train_categories = categorical(train_class, 1:6, ...
    {'Normal', 'MI', 'LBBB', 'RBBB', 'SB', 'AF'});
val_categories = categorical(val_class, 1:6, ...
    {'Normal', 'MI', 'LBBB', 'RBBB', 'SB', 'AF'});
test_categories = categorical(test_class, 1:6, ...
    {'Normal', 'MI', 'LBBB', 'RBBB', 'SB', 'AF'});

fprintf('✓ Labels converted to categorical\n\n');

%% STEP 6: Build Neural Network
fprintf('STEP 6: BUILD CNN ARCHITECTURE\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('6.1 Creating CNN1D architecture...\n');

layers = [
    sequenceInputLayer(12, 'Name', 'input')
    
    convolution1dLayer(50, 32, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling1dLayer(4, 'Stride', 4, 'Name', 'pool1')
    
    convolution1dLayer(30, 64, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling1dLayer(4, 'Stride', 4, 'Name', 'pool2')
    
    convolution1dLayer(20, 128, 'Padding', 'same', 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    maxPooling1dLayer(4, 'Stride', 4, 'Name', 'pool3')
    
    flattenLayer('Name', 'flatten')
    
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.5, 'Name', 'dropout1')
    
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu5')
    dropoutLayer(0.5, 'Name', 'dropout2')
    
    fullyConnectedLayer(6, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationOutputLayer('Name', 'output')
];

fprintf('✓ Network architecture created\n');
fprintf('  Layers: %d\n\n', length(layers));

%% STEP 7: Training Options
fprintf('STEP 7: TRAINING CONFIGURATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('7.1 Setting up training options...\n');

% Transpose data for trainNetwork (samples × time × features)
train_data_T = permute(train_data, [3 1 2]);
val_data_T = permute(val_data, [3 1 2]);
test_data_T = permute(test_data, [3 1 2]);

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationData', {val_data_T, val_categories}, ...
    'ValidationFrequency', 5, ...
    'ValidationPatience', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

fprintf('✓ Training options configured\n');
fprintf('  Epochs: 50\n');
fprintf('  Batch size: 8\n');
fprintf('  Optimizer: Adam\n');
fprintf('  Learning rate: 0.001\n\n');

%% STEP 8: Train Network
fprintf('STEP 8: TRAINING CNN\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

fprintf('8.1 Starting training...\n');
fprintf('(This may take 2-5 minutes)\n\n');

% Train the network
net = trainNetwork(train_data_T, train_categories, layers, options);

fprintf('\n✓ Training complete!\n\n');

%% STEP 9: Evaluate on Test Set
fprintf('STEP 9: TESTING\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

fprintf('9.1 Evaluating on test set...\n');

% Make predictions
test_pred = classify(net, test_data_T);

% Calculate accuracy
accuracy = sum(test_pred == test_categories) / length(test_categories);

fprintf('✓ Test accuracy: %.2f%%\n\n', accuracy * 100);

%% STEP 10: Confusion Matrix
fprintf('STEP 10: CONFUSION MATRIX\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('10.1 Computing confusion matrix...\n\n');

cm = confusionmat(test_categories, test_pred);
class_names = {'Normal', 'MI', 'LBBB', 'RBBB', 'SB', 'AF'};

fprintf('Confusion Matrix:\n');
fprintf('           Normal  MI  LBBB RBBB  SB   AF\n');
for i = 1:6
    fprintf('%s: ', class_names{i});
    fprintf('%5d ', cm(i, :));
    fprintf('\n');
end

% Calculate per-class accuracy
fprintf('\nPer-class Accuracy:\n');
for i = 1:6
    if sum(cm(i, :)) > 0
        class_acc = cm(i, i) / sum(cm(i, :));
        fprintf('  %s: %.2f%%\n', class_names{i}, class_acc * 100);
    end
end

fprintf('\n');

%% STEP 11: Save Results
fprintf('STEP 11: SAVING RESULTS\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('11.1 Saving trained network...\n');

save_path = fullfile(pwd, 'ecg_cnn_trained.mat');
save(save_path, 'net', '-v7.3');

fprintf('✓ Model saved to: %s\n\n', save_path);

%% STEP 12: Summary
fprintf('\n╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    TRAINING SUMMARY                           ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('Dataset:\n');
fprintf('  Total samples: %d\n', size(augmented_data, 3));
fprintf('  Training: %d | Validation: %d | Testing: %d\n\n', ...
    size(train_data, 3), size(val_data, 3), size(test_data, 3));

fprintf('Network:\n');
fprintf('  Type: 1D CNN\n');
fprintf('  Layers: %d\n', length(layers));
fprintf('  Parameters: ~250k\n\n');

fprintf('Training:\n');
fprintf('  Epochs: 50\n');
fprintf('  Batch size: 8\n');
fprintf('  Optimizer: Adam\n');
fprintf('  Learning rate: 0.001\n\n');

fprintf('Results:\n');
fprintf('  Test Accuracy: %.2f%%\n\n', accuracy * 100);

fprintf('Saved Files:\n');
fprintf('  ✓ ecg_cnn_trained.mat\n\n');

fprintf('════════════════════════════════════════════════════════════════\n\n');

fprintf('✅ TRAINING COMPLETE!\n\n');

fprintf('Next steps:\n');
fprintf('  1. Experiment with different architectures\n');
fprintf('  2. Try hyperparameter tuning\n');
fprintf('  3. Load real PTB Database data\n');
fprintf('  4. Deploy model to production\n\n');
