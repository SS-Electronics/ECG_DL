%% ECG CNN CLASSIFICATION - COMPLETE TRAINING WITH DEEP LEARNING TOOLBOX
% Full implementation with actual neural network training
% Uses modular load_ptb_database.m for clean data loading

clear; clc; close all;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  ECG CNN Classification - Full Training Pipeline              ║\n');
fprintf('║  WITH Deep Learning Toolbox + Modular Data Loading            ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% STEP 1: Initialize Project
fprintf('STEP 1: PROJECT INITIALIZATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

ProjectConfig.initialize();
fprintf('✓ Configuration loaded\n\n');

%% STEP 2: Load PTB Database (modular function)
fprintf('STEP 2: DATA LOADING\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

ptb_loader = PTBDataLoader(...
    ProjectConfig.PTB_DB_PATH, ...
    ProjectConfig.SAMPLING_RATE, ...
    ProjectConfig.SIGNAL_LENGTH, ...
    ProjectConfig.NUM_LEADS);

% ══════════════════════════════════════════════════════════════════
%  Single clean function call replaces the entire loading loop
%  Options:
%    load_ptb_database(path, loader)                    — load all
%    load_ptb_database(path, loader, 'MaxPatients', 100) — load 100
%    load_ptb_database(path, loader, 'Filter', false)    — skip filter
% ══════════════════════════════════════════════════════════════════

[all_data, all_labels, load_info] = load_ptb_database(...
    ProjectConfig.PTB_DB_PATH, ptb_loader, ...
    'MaxPatients', Inf, ...   % Load ALL patients (change to 100, 50, etc.)
    'Filter', true, ...       % Apply bandpass filter
    'Verbose', true);         % Print progress

% Sanity check
if load_info.loaded < 6
    error('Only %d patients loaded. Need at least 6 for 6-class classification.', load_info.loaded);
end

%% STEP 3: Class-Balanced Augmentation
fprintf('STEP 3: CLASS-BALANCED AUGMENTATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

n_samples = size(all_data, 3);
[~, class_ids] = max(all_labels, [], 2);
class_counts = histcounts(class_ids, 1:7);
max_count = max(class_counts);

fprintf('  Class distribution before augmentation:\n');
class_names = {'Normal', 'MI', 'LBBB', 'RBBB', 'SB', 'AF'};
for c = 1:6
    fprintf('    %-8s: %d samples\n', class_names{c}, class_counts(c));
end
fprintf('\n');

augmented_data = all_data;
augmented_labels = all_labels;

for sample = 1:n_samples
    cls = class_ids(sample);
    % More augmentations for minority classes, fewer for majority
    n_aug = max(1, round(max_count / max(class_counts(cls), 1)));
    
    for a = 1:n_aug
        signal = all_data(:, :, sample);
        % Random scaling (±10%)
        signal = signal * (0.90 + rand() * 0.20);
        % Gaussian noise
        signal = signal + randn(size(signal)) * 0.02;
        % Time shift (±200 samples)
        shift = randi([-200, 200]);
        signal = circshift(signal, shift, 1);
        
        augmented_data = cat(3, augmented_data, signal);
        augmented_labels = [augmented_labels; all_labels(sample, :)];
    end
end

% Print post-augmentation distribution
[~, aug_class_ids] = max(augmented_labels, [], 2);
aug_counts = histcounts(aug_class_ids, 1:7);
fprintf('  Class distribution after augmentation:\n');
for c = 1:6
    fprintf('    %-8s: %d samples\n', class_names{c}, aug_counts(c));
end
fprintf('\n  Total: %d → %d samples\n\n', n_samples, size(augmented_data, 3));

%% STEP 4: Data Split
fprintf('STEP 4: DATA SPLITTING\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

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

[~, train_class] = max(train_labels, [], 2);
[~, val_class] = max(val_labels, [], 2);
[~, test_class] = max(test_labels, [], 2);

train_categories = categorical(train_class, 1:6, class_names);
val_categories = categorical(val_class, 1:6, class_names);
test_categories = categorical(test_class, 1:6, class_names);

fprintf('✓ Labels converted to categorical\n\n');

%% STEP 6: Build Neural Network (with batch normalization)
fprintf('STEP 6: BUILD CNN ARCHITECTURE\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

layers = [
    sequenceInputLayer(12, 'Name', 'input', 'MinLength', 10000)
    
    convolution1dLayer(50, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling1dLayer(4, 'Stride', 4, 'Name', 'pool1')
    
    convolution1dLayer(30, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling1dLayer(4, 'Stride', 4, 'Name', 'pool2')
    
    convolution1dLayer(20, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling1dLayer(4, 'Stride', 4, 'Name', 'pool3')
    
    globalAveragePooling1dLayer('Name', 'gap')
    
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.3, 'Name', 'dropout1')
    
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu5')
    dropoutLayer(0.3, 'Name', 'dropout2')
    
    fullyConnectedLayer(6, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
];

fprintf('✓ Network architecture created (with BatchNorm)\n');
fprintf('  Layers: %d\n\n', length(layers));

%% STEP 7: Training Configuration
fprintf('STEP 7: TRAINING CONFIGURATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

% Convert to cell arrays for datastore
train_data_T = cell(size(train_data, 3), 1);
for i = 1:size(train_data, 3)
    train_data_T{i} = train_data(:, :, i);
end

val_data_T = cell(size(val_data, 3), 1);
for i = 1:size(val_data, 3)
    val_data_T{i} = val_data(:, :, i);
end

test_data_T = cell(size(test_data, 3), 1);
for i = 1:size(test_data, 3)
    test_data_T{i} = test_data(:, :, i);
end

% Wrap in datastores
trainDS = combine( ...
    arrayDatastore(train_data_T, 'OutputType', 'same'), ...
    arrayDatastore(train_categories));

valDS = combine( ...
    arrayDatastore(val_data_T, 'OutputType', 'same'), ...
    arrayDatastore(val_categories));

fprintf('✓ Datastores created\n');
fprintf('  Train: %d | Val: %d | Test: %d\n\n', ...
    length(train_data_T), length(val_data_T), length(test_data_T));

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationData', valDS, ...
    'ValidationFrequency', 10, ...
    'ValidationPatience', 15, ...
    'ExecutionEnvironment', 'gpu', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

fprintf('✓ Training options configured\n');
fprintf('  Epochs: 100 | Batch: 32 | LR: 0.001 | GPU: on\n\n');

%% STEP 8: Train Network
fprintf('STEP 8: TRAINING CNN\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

fprintf('Starting training...\n\n');
net = trainnet(trainDS, layers, "crossentropy", options);
fprintf('\n✓ Training complete!\n\n');

%% STEP 9: Evaluate on Test Set
fprintf('STEP 9: TESTING\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

testDS = arrayDatastore(test_data_T, 'OutputType', 'same');
scores = minibatchpredict(net, testDS);
test_pred = scores2label(scores, categories(train_categories));

accuracy = sum(test_pred == test_categories) / length(test_categories);
fprintf('✓ Test accuracy: %.2f%%\n\n', accuracy * 100);

%% STEP 10: Confusion Matrix
fprintf('STEP 10: CONFUSION MATRIX\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

cm = confusionmat(test_categories, test_pred);

fprintf('Confusion Matrix:\n');
fprintf('           Normal   MI  LBBB  RBBB    SB    AF\n');
for i = 1:6
    fprintf('%-8s', class_names{i});
    fprintf('%6d', cm(i, :));
    fprintf('\n');
end

fprintf('\nPer-class Accuracy:\n');
for i = 1:6
    if sum(cm(i, :)) > 0
        class_acc = cm(i, i) / sum(cm(i, :));
        fprintf('  %-8s: %5.1f%%  (%d/%d)\n', class_names{i}, ...
            class_acc * 100, cm(i,i), sum(cm(i,:)));
    else
        fprintf('  %-8s: N/A    (no test samples)\n', class_names{i});
    end
end
fprintf('\n');

%% STEP 11: Save Results
fprintf('STEP 11: SAVING RESULTS\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

save_path = fullfile(pwd, 'ecg_cnn_trained.mat');
save(save_path, 'net', 'load_info', 'accuracy', 'cm', 'class_names', '-v7.3');
fprintf('✓ Saved: %s\n\n', save_path);

%% STEP 12: Summary
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    TRAINING SUMMARY                           ║\n');
fprintf('╠════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Dataset: %d patients loaded, %d after augmentation          ║\n', ...
    load_info.loaded, size(augmented_data, 3));
fprintf('║  Split: %d train / %d val / %d test                          ║\n', ...
    size(train_data, 3), size(val_data, 3), size(test_data, 3));
fprintf('║  Network: 1D CNN + BatchNorm, 22 layers, ~311k params       ║\n');
fprintf('║  Training: 100 epochs, batch 32, Adam, GPU                   ║\n');
fprintf('║  Test Accuracy: %.2f%%                                       ║\n', accuracy * 100);
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('✅ TRAINING COMPLETE!\n\n');