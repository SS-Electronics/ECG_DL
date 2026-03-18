%% Advanced Training Pipeline Function
function [net, training_history, best_config] = training_pipeline(X_train, y_train, X_val, y_val, config)
    %TRAINING_PIPELINE Advanced training with validation and early stopping
    %
    % Inputs:
    %   X_train - Training data [leads x samples x segments]
    %   y_train - Training labels
    %   X_val - Validation data
    %   y_val - Validation labels
    %   config - Configuration structure
    %
    % Outputs:
    %   net - Trained neural network
    %   training_history - Training metrics history
    %   best_config - Configuration used for best model
    
    fprintf('\n========================================\n');
    fprintf('ADVANCED TRAINING PIPELINE\n');
    fprintf('========================================\n\n');
    
    % Create network architecture
    layers = create_ecg_cnn_layers(config);
    
    % Training options with advanced settings
    options = trainingOptions(config.optimizer, ...
        'InitialLearnRate', config.learning_rate, ...
        'MaxEpochs', config.max_epochs, ...
        'MiniBatchSize', config.batch_size, ...
        'ValidationData', {X_val, y_val}, ...
        'ValidationFrequency', 30, ...
        'Plots', 'training-progress', ...
        'Verbose', 1, ...
        'ExecutionEnvironment', 'auto', ...
        'GradientThresholdMethod', 'l2norm', ...
        'GradientThreshold', 1);
    
    % Early stopping
    if config.early_stopping
        options.ValidationPatience = config.early_stopping_patience;
    end
    
    % Train network
    fprintf('Starting training...\n');
    tic;
    [net, info] = trainNetwork(X_train, y_train, layers, options);
    training_time = toc;
    
    fprintf('Training completed in %.2f seconds\n', training_time);
    
    % Prepare output
    training_history = info;
    best_config = config;
    best_config.training_time = training_time;
end

%% Hyperparameter Tuning Function
function results = hyperparameter_tuning(X_train, y_train, X_val, y_val, config)
    %HYPERPARAMETER_TUNING Grid search for optimal hyperparameters
    %
    % Tests different combinations of learning rates and batch sizes
    %
    % Outputs:
    %   results - Structure with results for each configuration
    
    % Hyperparameter ranges
    learning_rates = [0.001, 0.0005, 0.0001];
    batch_sizes = [16, 32, 64];
    
    results = struct();
    result_idx = 1;
    
    for lr = learning_rates
        for bs = batch_sizes
            fprintf('\n--- Testing LR=%.5f, BS=%d ---\n', lr, bs);
            
            % Update config
            config_test = config;
            config_test.learning_rate = lr;
            config_test.batch_size = bs;
            config_test.max_epochs = 20;  % Short training for tuning
            
            % Train
            [net, info, ~] = training_pipeline(X_train, y_train, X_val, y_val, config_test);
            
            % Evaluate
            y_pred = classify(net, X_val);
            metrics = compute_classification_metrics(y_val, y_pred);
            
            % Store results
            results.configs(result_idx) = config_test;
            results.metrics(result_idx) = metrics;
            results.training_info(result_idx) = info;
            results.learning_rates(result_idx) = lr;
            results.batch_sizes(result_idx) = bs;
            results.final_accuracy(result_idx) = metrics.accuracy;
            
            fprintf('  Validation Accuracy: %.4f\n', metrics.accuracy);
            
            result_idx = result_idx + 1;
        end
    end
    
    % Find best configuration
    [~, best_idx] = max(results.final_accuracy);
    results.best_config = results.configs(best_idx);
    results.best_accuracy = results.final_accuracy(best_idx);
    results.best_idx = best_idx;
    
    fprintf('\n========================================\n');
    fprintf('Best Configuration:\n');
    fprintf('  Learning Rate: %.5f\n', results.best_config.learning_rate);
    fprintf('  Batch Size: %d\n', results.best_config.batch_size);
    fprintf('  Accuracy: %.4f\n', results.best_accuracy);
    fprintf('========================================\n');
end

%% K-Fold Cross-Validation Function
function cv_results = k_fold_cross_validation(X, y, k_folds, config)
    %K_FOLD_CROSS_VALIDATION Perform k-fold cross-validation
    %
    % Inputs:
    %   X - Feature matrix [leads x samples x segments]
    %   y - Labels
    %   k_folds - Number of folds
    %   config - Configuration structure
    %
    % Outputs:
    %   cv_results - Cross-validation results
    
    fprintf('\n========================================\n');
    fprintf('K-FOLD CROSS-VALIDATION (%d folds)\n', k_folds);
    fprintf('========================================\n\n');
    
    % Shuffle data
    n_samples = size(X, 3);
    idx = randperm(n_samples);
    X = X(:, :, idx);
    y = y(idx);
    
    % Fold size
    fold_size = floor(n_samples / k_folds);
    
    % Store results
    cv_results.accuracies = [];
    cv_results.f1_scores = [];
    cv_results.confusion_matrices = {};
    cv_results.fold_nets = {};
    
    % Cross-validation
    for fold = 1:k_folds
        fprintf('Fold %d/%d...\n', fold, k_folds);
        
        % Create fold indices
        test_start = (fold-1) * fold_size + 1;
        if fold == k_folds
            test_end = n_samples;
        else
            test_end = fold * fold_size;
        end
        
        test_idx = test_start:test_end;
        train_idx = setdiff(1:n_samples, test_idx);
        
        % Split data
        X_fold_train = X(:, :, train_idx);
        y_fold_train = y(train_idx);
        X_fold_test = X(:, :, test_idx);
        y_fold_test = y(test_idx);
        
        % Further split training into train/val
        train_size = floor(0.8 * length(train_idx));
        X_fold_train_split = X_fold_train(:, :, 1:train_size);
        y_fold_train_split = y_fold_train(1:train_size);
        X_fold_val = X_fold_train(:, :, train_size+1:end);
        y_fold_val = y_fold_train(train_size+1:end);
        
        % Train network
        [net, ~, ~] = training_pipeline(X_fold_train_split, y_fold_train_split, ...
                                        X_fold_val, y_fold_val, config);
        
        % Evaluate
        y_pred = classify(net, X_fold_test);
        metrics = compute_classification_metrics(y_fold_test, y_pred);
        
        % Store results
        cv_results.accuracies = [cv_results.accuracies; metrics.accuracy];
        cv_results.f1_scores = [cv_results.f1_scores; metrics.f1_score];
        cv_results.confusion_matrices{fold} = metrics.confusion_matrix;
        cv_results.fold_nets{fold} = net;
        
        fprintf('  Fold Accuracy: %.4f\n', metrics.accuracy);
    end
    
    % Summary statistics
    cv_results.mean_accuracy = mean(cv_results.accuracies);
    cv_results.std_accuracy = std(cv_results.accuracies);
    cv_results.mean_f1 = mean(cv_results.f1_scores);
    cv_results.std_f1 = std(cv_results.f1_scores);
    
    fprintf('\n========================================\n');
    fprintf('Cross-Validation Summary:\n');
    fprintf('  Mean Accuracy: %.4f ± %.4f\n', cv_results.mean_accuracy, cv_results.std_accuracy);
    fprintf('  Mean F1-Score: %.4f ± %.4f\n', cv_results.mean_f1, cv_results.std_f1);
    fprintf('========================================\n\n');
end

%% Learning Rate Finder Function
function learning_rates = find_learning_rate(X_train, y_train, X_val, y_val, config)
    %FIND_LEARNING_RATE Find optimal learning rate range
    %
    % Implements learning rate range test
    % Trains network with exponentially increasing learning rates
    %
    % Returns:
    %   learning_rates - Tested learning rates and their performance
    
    fprintf('\n========================================\n');
    fprintf('LEARNING RATE RANGE FINDER\n');
    fprintf('========================================\n\n');
    
    % Learning rate range
    min_lr = 1e-5;
    max_lr = 1;
    n_iterations = 50;
    
    % Create layer list
    layers = create_ecg_cnn_layers(config);
    
    % Exponential learning rates
    lr_values = logspace(log10(min_lr), log10(max_lr), n_iterations);
    
    losses = [];
    
    for i = 1:length(lr_values)
        lr = lr_values(i);
        
        % Short training with this learning rate
        options = trainingOptions('adam', ...
            'InitialLearnRate', lr, ...
            'MaxEpochs', 1, ...
            'MiniBatchSize', config.batch_size, ...
            'Plots', 'none', ...
            'Verbose', 0);
        
        try
            [net, info] = trainNetwork(X_train, y_train, layers, options);
            loss = info.TrainingLoss(end);
            losses = [losses; loss];
            
            if mod(i, 10) == 0
                fprintf('LR: %.6f, Loss: %.6f\n', lr, loss);
            end
        catch
            % Skip if training fails
            losses = [losses; inf];
        end
    end
    
    % Prepare output
    learning_rates.lr_values = lr_values;
    learning_rates.losses = losses;
    
    % Find range with steepest descent
    [~, best_idx] = min(losses);
    learning_rates.suggested_lr = lr_values(max(1, best_idx - 5));
    
    % Plot
    figure('Position', [100 100 800 400]);
    semilogx(lr_values, losses, 'LineWidth', 2);
    xlabel('Learning Rate'); ylabel('Loss');
    title('Learning Rate Range Test');
    grid on;
    
    fprintf('\nSuggested Learning Rate: %.6f\n', learning_rates.suggested_lr);
end

%% Ensemble Training Function
function ensemble_results = train_ensemble(X_train, y_train, X_val, y_val, config, n_models)
    %TRAIN_ENSEMBLE Train multiple models and combine predictions
    %
    % Inputs:
    %   n_models - Number of models in ensemble
    %
    % Outputs:
    %   ensemble_results - Ensemble predictions and metrics
    
    if nargin < 6
        n_models = 5;
    end
    
    fprintf('\n========================================\n');
    fprintf('ENSEMBLE TRAINING (%d models)\n', n_models);
    fprintf('========================================\n\n');
    
    ensemble_results.models = {};
    ensemble_results.predictions = [];
    
    for i = 1:n_models
        fprintf('Training model %d/%d...\n', i, n_models);
        
        % Train with random seed variation
        config_variant = config;
        config_variant.random_seed = config.random_seed + i;
        rng(config_variant.random_seed);
        
        % Train network
        [net, ~, ~] = training_pipeline(X_train, y_train, X_val, y_val, config_variant);
        
        % Store model
        ensemble_results.models{i} = net;
        
        % Get predictions
        pred = classify(net, X_val);
        ensemble_results.predictions = [ensemble_results.predictions, double(pred)];
    end
    
    % Ensemble prediction (majority voting)
    ensemble_predictions = mode(ensemble_results.predictions, 2);
    ensemble_results.ensemble_pred = categorical(ensemble_predictions);
    
    % Evaluate ensemble
    ensemble_metrics = compute_classification_metrics(y_val, ensemble_results.ensemble_pred);
    ensemble_results.metrics = ensemble_metrics;
    
    fprintf('\nEnsemble Accuracy: %.4f\n', ensemble_metrics.accuracy);
end

%% Model Checkpointing Function
function save_checkpoint(net, config, metrics, epoch, checkpoint_dir)
    %SAVE_CHECKPOINT Save model checkpoint during training
    
    if ~exist(checkpoint_dir, 'dir')
        mkdir(checkpoint_dir);
    end
    
    filename = fullfile(checkpoint_dir, sprintf('checkpoint_epoch_%04d.mat', epoch));
    save(filename, 'net', 'config', 'metrics', 'epoch');
    
    fprintf('Checkpoint saved: %s\n', filename);
end

%% Load Best Model Function
function [net, best_metrics] = load_best_model(results_dir)
    %LOAD_BEST_MODEL Load the best performing model from results
    
    checkpoint_files = dir(fullfile(results_dir, 'checkpoint_*.mat'));
    
    if isempty(checkpoint_files)
        error('No checkpoint files found in %s', results_dir);
    end
    
    % Load all checkpoints and find best
    best_accuracy = -inf;
    best_file = '';
    
    for i = 1:length(checkpoint_files)
        load(fullfile(results_dir, checkpoint_files(i).name), 'metrics');
        if metrics.accuracy > best_accuracy
            best_accuracy = metrics.accuracy;
            best_file = checkpoint_files(i).name;
        end
    end
    
    % Load best model
    load(fullfile(results_dir, best_file), 'net', 'metrics');
    best_metrics = metrics;
    
    fprintf('Loaded best model: %s (Accuracy: %.4f)\n', best_file, best_accuracy);
end
