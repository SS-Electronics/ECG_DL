%% Example 1: Basic CNN Training
% ==================================
% Simplest workflow: Load data, train CNN, evaluate

% clear all; close all; clc;
% addpath(genpath('./'))
% 
% % Configuration
% config = load_config();
% 
% % Load data
% [ptb_data, ptb_labels] = load_ptb_database(config.ptb_path, 50);
% [ptb_data_proc, ~] = preprocess_signals(ptb_data, config);
% [X, y] = segment_signals(ptb_data_proc, ptb_labels, config);
% 
% % Split data
% [X_train, X_test, y_train, y_test, ~] = train_test_split(X, y, 0.8, 42);
% 
% % Train
% layers = create_ecg_cnn_layers(config);
% options = trainingOptions('adam', 'MaxEpochs', 50, 'MiniBatchSize', 32);
% net = trainNetwork(X_train, y_train, layers, options);
% 
% % Evaluate
% y_pred = classify(net, X_test);
% metrics = compute_classification_metrics(y_test, y_pred);
% fprintf('Accuracy: %.4f\n', metrics.accuracy);

%% Example 2: Hyperparameter Tuning
% ==================================
% Find optimal learning rate and batch size

% clear all; close all; clc;
% addpath(genpath('./'))
% 
% % Configuration
% config = load_config();
% 
% % Load and prepare data
% [ptb_data, ptb_labels] = load_ptb_database(config.ptb_path, 30);
% [ptb_data_proc, ~] = preprocess_signals(ptb_data, config);
% [X, y] = segment_signals(ptb_data_proc, ptb_labels, config);
% 
% % Create train/val/test split
% n = size(X, 3);
% idx = randperm(n);
% 
% % 70% train, 15% val, 15% test
% train_end = floor(0.7 * n);
% val_end = floor(0.85 * n);
% 
% X_train = X(:, :, idx(1:train_end));
% y_train = y(idx(1:train_end));
% X_val = X(:, :, idx(train_end+1:val_end));
% y_val = y(idx(train_end+1:val_end));
% X_test = X(:, :, idx(val_end+1:end));
% y_test = y(idx(val_end+1:end));
% 
% % Hyperparameter tuning
% results = hyperparameter_tuning(X_train, y_train, X_val, y_val, config);
% 
% % Train with best config
% [net, ~, best_config] = training_pipeline(X_train, y_train, X_val, y_val, results.best_config);
% 
% % Evaluate on test set
% y_pred = classify(net, X_test);
% final_metrics = compute_classification_metrics(y_test, y_pred);
% fprintf('\nFinal Test Accuracy: %.4f\n', final_metrics.accuracy);

%% Example 3: K-Fold Cross-Validation
% ====================================
% Robust evaluation using cross-validation

% clear all; close all; clc;
% addpath(genpath('./'))
% 
% % Configuration
% config = load_config();
% 
% % Load and prepare data
% [ptb_data, ptb_labels] = load_ptb_database(config.ptb_path, 50);
% [ptb_data_proc, ~] = preprocess_signals(ptb_data, config);
% [X, y] = segment_signals(ptb_data_proc, ptb_labels, config);
% 
% % Perform 5-fold cross-validation
% cv_results = k_fold_cross_validation(X, y, 5, config);
% 
% fprintf('\n========================================\n');
% fprintf('5-Fold CV Results:\n');
% fprintf('  Mean Accuracy: %.4f ± %.4f\n', cv_results.mean_accuracy, cv_results.std_accuracy);
% fprintf('  Mean F1-Score: %.4f ± %.4f\n', cv_results.mean_f1, cv_results.std_f1);
% fprintf('========================================\n');

%% Example 4: Learning Rate Finder
% =================================
% Automatically find good learning rate

% clear all; close all; clc;
% addpath(genpath('./'))
% 
% % Configuration
% config = load_config();
% 
% % Load and prepare data
% [ptb_data, ptb_labels] = load_ptb_database(config.ptb_path, 20);
% [ptb_data_proc, ~] = preprocess_signals(ptb_data, config);
% [X, y] = segment_signals(ptb_data_proc, ptb_labels, config);
% 
% % Split for learning rate finder
% [X_train, X_test, y_train, y_test, ~] = train_test_split(X, y, 0.8, 42);
% [X_train2, X_val, y_train2, y_val, ~] = train_test_split(X_train, y_train, 0.75, 42);
% 
% % Find learning rate
% lr_results = find_learning_rate(X_train2, y_train2, X_val, y_val, config);
% 
% % Use suggested learning rate for training
% config.learning_rate = lr_results.suggested_lr;
% [net, ~, ~] = training_pipeline(X_train, y_train, X_val, y_val, config);
% 
% % Evaluate
% y_pred = classify(net, X_test);
% metrics = compute_classification_metrics(y_test, y_pred);
% fprintf('Test Accuracy with optimized LR: %.4f\n', metrics.accuracy);

%% Example 5: Ensemble Learning
% =============================
% Train multiple models and combine predictions

% clear all; close all; clc;
% addpath(genpath('./'))
% 
% % Configuration
% config = load_config();
% 
% % Load and prepare data
% [ptb_data, ptb_labels] = load_ptb_database(config.ptb_path, 30);
% [ptb_data_proc, ~] = preprocess_signals(ptb_data, config);
% [X, y] = segment_signals(ptb_data_proc, ptb_labels, config);
% 
% % Split data
% [X_train, X_test, y_train, y_test, ~] = train_test_split(X, y, 0.8, 42);
% [X_train2, X_val, y_train2, y_val, ~] = train_test_split(X_train, y_train, 0.75, 42);
% 
% % Train ensemble of 5 models
% ensemble_results = train_ensemble(X_train2, y_train2, X_val, y_val, config, 5);
% 
% % Evaluate ensemble on test set
% test_preds = [];
% for i = 1:length(ensemble_results.models)
%     pred = classify(ensemble_results.models{i}, X_test);
%     test_preds = [test_preds, double(pred)];
% end
% 
% % Majority voting
% ensemble_test_pred = categorical(mode(test_preds, 2));
% ensemble_test_metrics = compute_classification_metrics(y_test, ensemble_test_pred);
% 
% fprintf('Ensemble Test Accuracy: %.4f\n', ensemble_test_metrics.accuracy);

%% Example 6: Different Architecture Comparison
% =============================================
% Compare standard CNN with ResNet and Attention

% clear all; close all; clc;
% addpath(genpath('./'))
% 
% % Configuration
% config = load_config();
% 
% % Load and prepare data
% [ptb_data, ptb_labels] = load_ptb_database(config.ptb_path, 40);
% [ptb_data_proc, ~] = preprocess_signals(ptb_data, config);
% [X, y] = segment_signals(ptb_data_proc, ptb_labels, config);
% 
% % Split data
% [X_train, X_test, y_train, y_test, ~] = train_test_split(X, y, 0.8, 42);
% [X_train2, X_val, y_train2, y_val, ~] = train_test_split(X_train, y_train, 0.75, 42);
% 
% results = struct();
% 
% % Standard CNN
% fprintf('Training Standard CNN...\n');
% layers_standard = create_ecg_cnn_layers(config);
% config_test = config;
% config_test.max_epochs = 30;
% [net_standard, ~, ~] = training_pipeline(X_train2, y_train2, X_val, y_val, config_test);
% y_pred_standard = classify(net_standard, X_test);
% results.standard = compute_classification_metrics(y_test, y_pred_standard);
% 
% % ResNet
% fprintf('Training ResNet-inspired CNN...\n');
% layers_resnet = create_ecg_cnn_resnet(config);
% [net_resnet, ~, ~] = training_pipeline(X_train2, y_train2, X_val, y_val, config_test);
% y_pred_resnet = classify(net_resnet, X_test);
% results.resnet = compute_classification_metrics(y_test, y_pred_resnet);
% 
% % Attention CNN
% fprintf('Training Attention-based CNN...\n');
% layers_attention = create_ecg_cnn_attention(config);
% [net_attention, ~, ~] = training_pipeline(X_train2, y_train2, X_val, y_val, config_test);
% y_pred_attention = classify(net_attention, X_test);
% results.attention = compute_classification_metrics(y_test, y_pred_attention);
% 
% % Compare
% fprintf('\n========================================\n');
% fprintf('Architecture Comparison:\n');
% fprintf('  Standard CNN:      %.4f\n', results.standard.accuracy);
% fprintf('  ResNet:            %.4f\n', results.resnet.accuracy);
% fprintf('  Attention:         %.4f\n', results.attention.accuracy);
% fprintf('========================================\n');

%% Example 7: Full Production Pipeline
% ====================================
% Complete workflow with all optimizations

% clear all; close all; clc;
% addpath(genpath('./'))
% 
% % Load configuration
% config = load_config();
% config.max_patients = 100;
% 
% fprintf('Starting Production Pipeline...\n\n');
% 
% % Stage 1: Data Loading
% fprintf('Stage 1: Data Loading\n');
% [ptb_data, ptb_labels] = load_ptb_database(config.ptb_path, config.max_patients);
% 
% % Stage 2: Preprocessing
% fprintf('\nStage 2: Preprocessing\n');
% [ptb_data_proc, preprocess_info] = preprocess_signals(ptb_data, config);
% 
% % Stage 3: Segmentation
% fprintf('\nStage 3: Segmentation\n');
% [X, y] = segment_signals(ptb_data_proc, ptb_labels, config);
% 
% % Stage 4: Data Split
% fprintf('\nStage 4: Data Splitting\n');
% [X_train, X_test, y_train, y_test, ~] = train_test_split(X, y, 0.8, 42);
% [X_train, X_val, y_train, y_val, ~] = train_test_split(X_train, y_train, 0.75, 42);
% 
% % Stage 5: Learning Rate Optimization
% fprintf('\nStage 5: Learning Rate Finder\n');
% lr_results = find_learning_rate(X_train, y_train, X_val, y_val, config);
% config.learning_rate = lr_results.suggested_lr;
% 
% % Stage 6: Model Training
% fprintf('\nStage 6: Model Training\n');
% [net, train_info, best_config] = training_pipeline(X_train, y_train, X_val, y_val, config);
% 
% % Stage 7: Evaluation
% fprintf('\nStage 7: Evaluation\n');
% y_pred = classify(net, X_test);
% metrics = compute_classification_metrics(y_test, y_pred);
% 
% % Stage 8: Save Results
% fprintf('\nStage 8: Saving Results\n');
% if ~exist('results', 'dir')
%     mkdir('results');
% end
% 
% save('results/final_model.mat', 'net', 'config', 'metrics', 'train_info');
% fprintf('Model saved to results/final_model.mat\n');
% 
% % Summary
% fprintf('\n========================================\n');
% fprintf('PRODUCTION PIPELINE COMPLETED\n');
% fprintf('========================================\n');
% fprintf('Test Set Performance:\n');
% fprintf('  Accuracy:  %.4f\n', metrics.accuracy);
% fprintf('  Precision: %.4f\n', metrics.precision);
% fprintf('  Recall:    %.4f\n', metrics.recall);
% fprintf('  F1-Score:  %.4f\n', metrics.f1_score);
% fprintf('========================================\n');

%% Script Notes
% =============
% 1. Uncomment one example at a time to run
% 2. Adjust config.max_patients based on available memory
% 3. Ensure all paths are correctly set in load_config.m
% 4. Use smaller datasets for quick testing
% 5. Monitor GPU/CPU usage and memory consumption
% 6. Save important results before closing MATLAB
