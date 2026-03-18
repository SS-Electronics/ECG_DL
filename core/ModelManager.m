classdef ModelManager < handle
    % ModelManager - Manages multiple model architectures and training
    % Handles model registration, selection, and lifecycle management
    
    properties
        available_models    % Cell array of available model classes
        trained_models      % Struct array of trained models
        model_registry      % Registry of model configurations
        config              % Project configuration
    end
    
    methods
        function obj = ModelManager(config)
            % Initialize ModelManager
            obj.config = config;
            obj.available_models = {};
            obj.trained_models = struct();
            obj.model_registry = struct();
            obj.registerDefaultModels();
        end
        
        function registerDefaultModels(obj)
            % Register available model architectures
            % These are the models that can be trained
            
            % Model 1: CNN1D
            obj.registerModel('CNN1D', 'Standard 1D Convolutional Neural Network', ...
                'CNN1D', obj.config.DEFAULT_MODEL_PARAMS);
            
            % Model 2: ResNetECG
            obj.registerModel('ResNetECG', 'ResNet-inspired architecture with residual connections', ...
                'ResNetECG', obj.config.DEFAULT_MODEL_PARAMS);
            
            fprintf('[ModelManager] %d models registered\n', length(obj.available_models));
        end
        
        function registerModel(obj, model_name, description, class_name, params)
            % Register a new model architecture
            % model_name: Unique identifier (e.g., 'CNN1D')
            % description: Human-readable description
            % class_name: Name of the MATLAB class
            % params: Default parameters struct
            
            idx = length(obj.available_models) + 1;
            obj.available_models{idx} = model_name;
            
            obj.model_registry.(model_name).description = description;
            obj.model_registry.(model_name).class_name = class_name;
            obj.model_registry.(model_name).default_params = params;
            obj.model_registry.(model_name).registration_date = datetime('now');
            
            fprintf('[ModelManager] Registered model: %s\n', model_name);
        end
        
        function model = createModel(obj, model_name, custom_params)
            % Create an instance of a registered model
            % model_name: Name of registered model
            % custom_params: (optional) Override default parameters
            
            if ~ismember(model_name, obj.available_models)
                error('[ModelManager] Model not found: %s', model_name);
            end
            
            class_name = obj.model_registry.(model_name).class_name;
            params = obj.model_registry.(model_name).default_params;
            
            % Override with custom parameters
            if nargin > 2 && ~isempty(custom_params)
                params = obj.mergeParams(params, custom_params);
            end
            
            % Create model instance
            model = feval(class_name, params);
            
            fprintf('[ModelManager] Created model: %s\n', model_name);
        end
        
        function model = trainModel(obj, model_name, train_data, train_labels, ...
                                     val_data, val_labels, training_config)
            % Train a model
            
            % Create model
            model = obj.createModel(model_name);
            
            % Set up training configuration
            if nargin < 7
                training_config = struct(...
                    'num_epochs', obj.config.NUM_EPOCHS, ...
                    'batch_size', obj.config.BATCH_SIZE, ...
                    'learning_rate', obj.config.LEARNING_RATE, ...
                    'patience', obj.config.VALIDATION_PATIENCE);
            end
            
            % Train the model
            fprintf('[ModelManager] Training %s...\n', model_name);
            model.train(train_data, train_labels, val_data, val_labels, training_config);
            
            % Store trained model
            obj.trained_models.(model_name) = model;
            
            fprintf('[ModelManager] Model %s training completed\n', model_name);
        end
        
        function [predictions, confidence] = predictWithModel(obj, model_name, input_data)
            % Make predictions with a trained model
            
            if ~isfield(obj.trained_models, model_name)
                error('[ModelManager] Model not trained: %s', model_name);
            end
            
            model = obj.trained_models.(model_name);
            predictions = predict(model, input_data);
            
            % Get confidence scores and class labels
            [confidence, class_idx] = max(predictions, [], 2);
            predictions = class_idx;
        end
        
        function saveModel(obj, model_name, save_path)
            % Save a trained model
            
            if ~isfield(obj.trained_models, model_name)
                error('[ModelManager] Model not trained: %s', model_name);
            end
            
            model = obj.trained_models.(model_name);
            full_path = fullfile(save_path, [model_name '_trained.mat']);
            saveModel(model, full_path);
        end
        
        function loadModel(obj, model_name, load_path)
            % Load a trained model
            
            full_path = fullfile(load_path, [model_name '_trained.mat']);
            
            if ~isfile(full_path)
                error('[ModelManager] Model file not found: %s', full_path);
            end
            
            loaded = load(full_path);
            obj.trained_models.(model_name) = loaded.obj;
            
            fprintf('[ModelManager] Loaded model: %s\n', model_name);
        end
        
        function listAvailableModels(obj)
            % Display all available models
            fprintf('\n===== Available Models =====\n');
            for i = 1:length(obj.available_models)
                model_name = obj.available_models{i};
                info = obj.model_registry.(model_name);
                fprintf('%d. %s\n', i, model_name);
                fprintf('   Description: %s\n', info.description);
                fprintf('   Class: %s\n', info.class_name);
                fprintf('   Registered: %s\n', char(info.registration_date));
            end
            fprintf('============================\n\n');
        end
        
        function listTrainedModels(obj)
            % Display all trained models
            trained = fieldnames(obj.trained_models);
            
            if isempty(trained)
                fprintf('[ModelManager] No trained models yet\n');
                return;
            end
            
            fprintf('\n===== Trained Models =====\n');
            for i = 1:length(trained)
                model_name = trained{i};
                model = obj.trained_models.(model_name);
                fprintf('%d. %s\n', i, model_name);
                fprintf('   Is trained: %s\n', bool2str(model.is_trained));
                fprintf('   Epochs: %d\n', length(model.training_history.epoch));
                if ~isempty(model.training_history.val_acc)
                    fprintf('   Best Val Acc: %.4f\n', max(model.training_history.val_acc));
                end
            end
            fprintf('==========================\n\n');
        end
        
        function compareModels(obj, model_names)
            % Compare multiple trained models
            % model_names: Cell array of model names to compare
            
            fprintf('\n===== Model Comparison =====\n');
            fprintf('%-20s %-15s %-15s %-15s\n', 'Model', 'Final Train Acc', 'Final Val Acc', 'Best Val Acc');
            fprintf('%s\n', repmat('-', 1, 65));
            
            for i = 1:length(model_names)
                model_name = model_names{i};
                if isfield(obj.trained_models, model_name)
                    model = obj.trained_models.(model_name);
                    train_acc = model.training_history.train_acc(end);
                    val_acc = model.training_history.val_acc(end);
                    best_val_acc = max(model.training_history.val_acc);
                    fprintf('%-20s %-15.4f %-15.4f %-15.4f\n', ...
                        model_name, train_acc, val_acc, best_val_acc);
                else
                    fprintf('%-20s %-15s %-15s %-15s\n', model_name, 'N/A', 'N/A', 'N/A');
                end
            end
            fprintf('============================\n\n');
        end
        
        function plotComparison(obj, model_names)
            % Plot training curves for multiple models
            
            figure('Name', 'Model Comparison', 'NumberTitle', 'off');
            
            colors = lines(length(model_names));
            
            % Loss comparison
            subplot(1, 2, 1);
            for i = 1:length(model_names)
                if isfield(obj.trained_models, model_names{i})
                    model = obj.trained_models.(model_names{i});
                    plot(model.training_history.epoch, model.training_history.val_loss, ...
                        'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', model_names{i});
                    hold on;
                end
            end
            xlabel('Epoch');
            ylabel('Validation Loss');
            title('Validation Loss Comparison');
            legend;
            grid on;
            
            % Accuracy comparison
            subplot(1, 2, 2);
            for i = 1:length(model_names)
                if isfield(obj.trained_models, model_names{i})
                    model = obj.trained_models.(model_names{i});
                    plot(model.training_history.epoch, model.training_history.val_acc, ...
                        'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', model_names{i});
                    hold on;
                end
            end
            xlabel('Epoch');
            ylabel('Validation Accuracy');
            title('Validation Accuracy Comparison');
            legend;
            grid on;
        end
        
        function merged_params = mergeParams(obj, default_params, custom_params)
            % Merge custom parameters with defaults
            merged_params = default_params;
            custom_fields = fieldnames(custom_params);
            for i = 1:length(custom_fields)
                field = custom_fields{i};
                merged_params.(field) = custom_params.(field);
            end
        end
    end
end

function str = bool2str(val)
    if val
        str = 'Yes';
    else
        str = 'No';
    end
end
