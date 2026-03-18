%% SIMPLE ECG CNN TRAINING - WORKING VERSION
% This version generates proper synthetic ECG data
% No dimension mismatch errors!

clear; clc; close all;

fprintf('\n╔═══════════════════════════════════════════════════════════════╗\n');
fprintf('║   ECG CNN Classification - Simple Training Pipeline         ║\n');
fprintf('╚═══════════════════════════════════════════════════════════════╝\n\n');

%% STEP 1: Initialize
fprintf('STEP 1: PROJECT INITIALIZATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

ProjectConfig.initialize();

signal_length = 10000;
num_leads = 12;
n_samples = 30;

fprintf('✓ Project initialized\n');
fprintf('  Signal length: %d samples\n', signal_length);
fprintf('  Number of leads: %d\n', num_leads);
fprintf('  Num samples: %d\n\n', n_samples);

%% STEP 2: Generate synthetic ECG data
fprintf('STEP 2: GENERATE SYNTHETIC ECG DATA\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('2.1 Creating synthetic ECG signals...\n');

all_data = zeros(signal_length, num_leads, n_samples);
all_labels = zeros(n_samples, 6);

% Time vector as column vector
t = (0:signal_length-1)' / 1000;  % Time in seconds (10000 x 1)

for sample = 1:n_samples
    % Cycle through disease classes
    disease_class = mod(sample-1, 6) + 1;
    
    % Initialize signal with noise
    signal = randn(signal_length, num_leads) * 0.1;
    
    % Create ECG-like patterns for each disease
    for lead = 1:num_leads
        if disease_class == 1  % Normal
            ecg = 0.8*sin(2*pi*1*t) + 0.3*sin(4*pi*1*t);
            
        elseif disease_class == 2  % MI
            ecg = 0.5*sin(2*pi*1*t) - 0.3;
            
        elseif disease_class == 3  % LBBB
            ecg = 1.2*sin(2*pi*1*t);
            
        elseif disease_class == 4  % RBBB
            ecg = 0.9*sin(2*pi*1*t) + 0.2*sin(6*pi*1*t);
            
        elseif disease_class == 5  % SB (slow)
            ecg = 0.7*sin(2*pi*0.67*t);  % Slower rate
            
        else  % AF (irregular)
            irregular_freq = 1 + 0.3*randn();
            ecg = 0.5*sin(2*pi*irregular_freq*t) + 0.2*randn(signal_length, 1);
        end
        
        % Add to signal
        signal(:, lead) = signal(:, lead) + ecg;
        
        % Normalize this lead
        signal(:, lead) = (signal(:, lead) - mean(signal(:, lead))) / (std(signal(:, lead)) + 1e-6);
    end
    
    % Store
    all_data(:, :, sample) = signal;
    all_labels(sample, disease_class) = 1;
end

fprintf('✓ Generated %d synthetic ECG samples\n', n_samples);
fprintf('  Shape: (%d, %d, %d)\n\n', size(all_data, 1), size(all_data, 2), size(all_data, 3));

%% STEP 3: Data augmentation
fprintf('STEP 3: DATA AUGMENTATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('3.1 Augmenting data (2x multiplication)...\n');

augmented_data = all_data;
for sample = 1:n_samples
    signal = all_data(:, :, sample);
    
    % Scale + noise
    signal = signal * (0.95 + rand() * 0.1);
    signal = signal + randn(size(signal)) * 0.01;
    
    augmented_data = cat(3, augmented_data, signal);
end

augmented_labels = repmat(all_labels, 2, 1);

fprintf('✓ Augmentation complete\n');
fprintf('  Dataset size: %d → %d samples\n\n', n_samples, size(augmented_data, 3));

%% STEP 4: Data split
fprintf('STEP 4: TRAIN/VAL/TEST SPLIT\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('4.1 Creating splits...\n');

n_total = size(augmented_data, 3);
n_train = floor(n_total * 0.6);
n_val = floor(n_total * 0.2);
n_test = n_total - n_train - n_val;

% Random permutation
idx = randperm(n_total);
idx_train = idx(1:n_train);
idx_val = idx(n_train+1:n_train+n_val);
idx_test = idx(n_train+n_val+1:end);

% Split data
train_data = augmented_data(:, :, idx_train);
val_data = augmented_data(:, :, idx_val);
test_data = augmented_data(:, :, idx_test);

% Split labels
train_labels = augmented_labels(idx_train, :);
val_labels = augmented_labels(idx_val, :);
test_labels = augmented_labels(idx_test, :);

fprintf('✓ Split complete\n');
fprintf('  Train: %d samples\n', size(train_data, 3));
fprintf('  Val:   %d samples\n', size(val_data, 3));
fprintf('  Test:  %d samples\n\n', size(test_data, 3));

%% STEP 5: Models
fprintf('STEP 5: MODEL INITIALIZATION\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

fprintf('5.1 Creating models...\n');

model_manager = ModelManager(ProjectConfig);
model1 = CNN1D(ProjectConfig.DEFAULT_MODEL_PARAMS);
model2 = ResNetECG(ProjectConfig.DEFAULT_MODEL_PARAMS);

fprintf('✓ CNN1D created\n');
fprintf('✓ ResNetECG created\n\n');

%% SUMMARY
fprintf('════════════════════════════════════════════════════════════════\n');
fprintf('✅ FRAMEWORK READY FOR TRAINING\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

fprintf('Dataset Summary:\n');
fprintf('  Total samples: %d\n', size(augmented_data, 3));
fprintf('  Train/Val/Test: %d / %d / %d\n', ...
    size(train_data, 3), size(val_data, 3), size(test_data, 3));
fprintf('  Classes: 6 diseases\n');
fprintf('  Signal length: %d samples (10 seconds)\n', signal_length);
fprintf('  Leads: %d-lead ECG\n\n', num_leads);

fprintf('Models Ready:\n');
fprintf('  ✓ CNN1D (3 conv blocks + 3 dense layers)\n');
fprintf('  ✓ ResNetECG (3 residual blocks)\n\n');

fprintf('Status: ✅ READY FOR DEEP LEARNING TRAINING\n');
fprintf('Next: Integrate trainNetwork() from Deep Learning Toolbox\n\n');

fprintf('════════════════════════════════════════════════════════════════\n\n');
