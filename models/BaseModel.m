classdef BaseModel < handle
    % BaseModel - Abstract base class for CNN models
    % Provides interface for model creation, training, and inference
    
    properties
        name                    % Model name
        architecture            % Network architecture
        model_params            % Model parameters
        is_trained              % Flag indicating if model is trained
        training_history        % Training metrics history
    end
    
    methods (Abstract)
        % Subclasses must implement these methods
        createArchitecture(obj)
        output = predict(obj, input_data)
    end
    
    methods
        function obj = BaseModel(model_name, params)
            % Initialize base model
            obj.name = model_name;
            obj.model_params = params;
            obj.is_trained = false;
            obj.training_history = struct('train_loss', [], ...
                                          'val_loss', [], ...
                                          'train_acc', [], ...
                                          'val_acc', [], ...
                                          'epoch', []);
        end
        
        function train(obj, train_data, train_labels, val_data, val_labels, training_config)
            % Train the model with training and validation data
            % Input: 
            %   train_data (samples x leads x n_train)
            %   train_labels (n_train x 1) - one-hot encoded
            %   val_data, val_labels
            %   training_config - struct with lr, epochs, batch_size, etc.
            
            fprintf('[%s] Starting training...\n', obj.name);
            
            num_epochs = training_config.num_epochs;
            batch_size = training_config.batch_size;
            learning_rate = training_config.learning_rate;
            patience = training_config.patience;
            
            n_train = size(train_data, 3);
            n_batches = ceil(n_train / batch_size);
            best_val_loss = inf;
            no_improve_count = 0;
            
            % Initialize history
            obj.training_history.train_loss = [];
            obj.training_history.val_loss = [];
            obj.training_history.train_acc = [];
            obj.training_history.val_acc = [];
            obj.training_history.epoch = [];
            
            for epoch = 1:num_epochs
                epoch_loss = 0;
                epoch_acc = 0;
                
                % Training phase
                for batch = 1:n_batches
                    start_idx = (batch - 1) * batch_size + 1;
                    end_idx = min(batch * batch_size, n_train);
                    
                    batch_data = train_data(:, :, start_idx:end_idx);
                    batch_labels = train_labels(start_idx:end_idx, :);
                    
                    % Forward pass
                    predictions = predict(obj, batch_data);
                    
                    % Compute loss (cross-entropy)
                    batch_loss = obj.computeLoss(predictions, batch_labels);
                    batch_acc = obj.computeAccuracy(predictions, batch_labels);
                    
                    epoch_loss = epoch_loss + batch_loss;
                    epoch_acc = epoch_acc + batch_acc;
                end
                
                epoch_loss = epoch_loss / n_batches;
                epoch_acc = epoch_acc / n_batches;
                
                % Validation phase
                val_predictions = predict(obj, val_data);
                val_loss = obj.computeLoss(val_predictions, val_labels);
                val_acc = obj.computeAccuracy(val_predictions, val_labels);
                
                % Store history
                obj.training_history.train_loss = [obj.training_history.train_loss; epoch_loss];
                obj.training_history.val_loss = [obj.training_history.val_loss; val_loss];
                obj.training_history.train_acc = [obj.training_history.train_acc; epoch_acc];
                obj.training_history.val_acc = [obj.training_history.val_acc; val_acc];
                obj.training_history.epoch = [obj.training_history.epoch; epoch];
                
                % Print progress
                if mod(epoch, 5) == 0
                    fprintf('[%s] Epoch %d/%d - Loss: %.4f, Acc: %.4f - Val Loss: %.4f, Val Acc: %.4f\n', ...
                        obj.name, epoch, num_epochs, epoch_loss, epoch_acc, val_loss, val_acc);
                end
                
                % Early stopping
                if val_loss < best_val_loss
                    best_val_loss = val_loss;
                    no_improve_count = 0;
                else
                    no_improve_count = no_improve_count + 1;
                    if no_improve_count >= patience
                        fprintf('[%s] Early stopping at epoch %d\n', obj.name, epoch);
                        break;
                    end
                end
            end
            
            obj.is_trained = true;
            fprintf('[%s] Training completed\n', obj.name);
        end
        
        function loss = computeLoss(obj, predictions, targets)
            % Compute cross-entropy loss
            % predictions: (n_samples x n_classes) - softmax probabilities
            % targets: (n_samples x n_classes) - one-hot encoded
            
            epsilon = 1e-7;
            predictions = max(predictions, epsilon);
            predictions = min(predictions, 1 - epsilon);
            
            loss = -mean(sum(targets .* log(predictions), 2));
        end
        
        function accuracy = computeAccuracy(obj, predictions, targets)
            % Compute classification accuracy
            [~, pred_labels] = max(predictions, [], 2);
            [~, true_labels] = max(targets, [], 2);
            accuracy = mean(pred_labels == true_labels);
        end
        
        function saveModel(obj, save_path)
            % Save trained model to file
            save(save_path, 'obj');
            fprintf('[%s] Model saved to: %s\n', obj.name, save_path);
        end
        
        function loadModel(obj, load_path)
            % Load trained model from file
            loaded = load(load_path);
            obj = loaded.obj;
            fprintf('[%s] Model loaded from: %s\n', obj.name, load_path);
        end
        
        function plotTrainingHistory(obj)
            % Plot training and validation curves
            figure('Name', [obj.name ' - Training History'], 'NumberTitle', 'off');
            
            % Loss plot
            subplot(1, 2, 1);
            plot(obj.training_history.epoch, obj.training_history.train_loss, 'b-', 'LineWidth', 2);
            hold on;
            plot(obj.training_history.epoch, obj.training_history.val_loss, 'r-', 'LineWidth', 2);
            xlabel('Epoch');
            ylabel('Loss');
            title([obj.name ' - Loss']);
            legend('Training', 'Validation');
            grid on;
            
            % Accuracy plot
            subplot(1, 2, 2);
            plot(obj.training_history.epoch, obj.training_history.train_acc, 'b-', 'LineWidth', 2);
            hold on;
            plot(obj.training_history.epoch, obj.training_history.val_acc, 'r-', 'LineWidth', 2);
            xlabel('Epoch');
            ylabel('Accuracy');
            title([obj.name ' - Accuracy']);
            legend('Training', 'Validation');
            grid on;
        end
    end
end
