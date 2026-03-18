%% Train-Test Split Function
function [X_train, X_test, y_train, y_test, info] = train_test_split(X, y, train_ratio, random_seed)
    %TRAIN_TEST_SPLIT Split data into training and testing sets
    %
    % Inputs:
    %   X - Feature matrix
    %   y - Label vector
    %   train_ratio - Training set ratio (0-1)
    %   random_seed - Random seed for reproducibility
    %
    % Outputs:
    %   X_train, X_test - Split feature matrices
    %   y_train, y_test - Split labels
    %   info - Split information
    
    % Set random seed for reproducibility
    rng(random_seed);
    
    % Get number of samples
    n_samples = size(X, 3);
    
    % Generate random permutation
    idx = randperm(n_samples);
    
    % Calculate split point
    split_point = floor(n_samples * train_ratio);
    
    % Split indices
    train_idx = idx(1:split_point);
    test_idx = idx(split_point+1:end);
    
    % Split data
    X_train = X(:, :, train_idx);
    X_test = X(:, :, test_idx);
    y_train = y(train_idx);
    y_test = y(test_idx);
    
    % Prepare info structure
    info.train_size = length(train_idx);
    info.test_size = length(test_idx);
    info.train_ratio = train_ratio;
    info.random_seed = random_seed;
    
    % Class distribution
    [train_counts, classes] = hist(y_train, unique(y_train));
    [test_counts, ~] = hist(y_test, unique(y_test));
    
    info.train_class_distribution = [string(classes)' num2cell(train_counts)];
    info.test_class_distribution = [string(classes)' num2cell(test_counts)];
end

%% Tabulate Function (for class distribution display)
function result = tabulate(data)
    %TABULATE Display frequency table of categorical data
    %
    % Returns:
    %   result - Table with value counts and percentages
    
    data_cat = categorical(data);
    [counts, categories] = hist(data_cat, unique(data_cat));
    percentages = (counts / length(data)) * 100;
    
    result = table(string(categories)', counts', percentages', ...
        'VariableNames', {'Category', 'Count', 'Percentage'});
end

%% Confusion Matrix Helper
function cm = confusionmat(y_true, y_pred)
    %CONFUSIONMAT Compute confusion matrix
    
    % Convert to numeric if categorical
    if iscategorical(y_true)
        y_true = double(y_true);
    end
    if iscategorical(y_pred)
        y_pred = double(y_pred);
    end
    
    % Get unique classes
    classes = unique([y_true; y_pred]);
    n_classes = length(classes);
    
    cm = zeros(n_classes, n_classes);
    
    for i = 1:n_classes
        for j = 1:n_classes
            cm(i, j) = sum((y_true == classes(i)) & (y_pred == classes(j)));
        end
    end
end

%% Standard Scaler for Features
function [X_scaled, scaler] = standardize_features(X, fit_data)
    %STANDARDIZE_FEATURES Standardize features to zero mean and unit variance
    %
    % Inputs:
    %   X - Input features
    %   fit_data (optional) - Data to fit scaler on
    %
    % Outputs:
    %   X_scaled - Scaled features
    %   scaler - Scaler parameters
    
    if nargin < 2
        fit_data = X;
    end
    
    % Calculate mean and std
    scaler.mean = mean(fit_data, 'all');
    scaler.std = std(fit_data, 0, 'all');
    
    % Avoid division by zero
    scaler.std(scaler.std == 0) = 1;
    
    % Scale data
    X_scaled = (X - scaler.mean) / scaler.std;
end

%% Min-Max Scaler for Features
function [X_scaled, scaler] = minmax_scale_features(X, fit_data)
    %MINMAX_SCALE_FEATURES Scale features to [0, 1] range
    
    if nargin < 2
        fit_data = X;
    end
    
    % Calculate min and max
    scaler.min = min(fit_data, [], 'all');
    scaler.max = max(fit_data, [], 'all');
    scaler.range = scaler.max - scaler.min;
    
    % Avoid division by zero
    scaler.range(scaler.range == 0) = 1;
    
    % Scale data
    X_scaled = (X - scaler.min) / scaler.range;
end

%% Class Weight Calculator (for imbalanced datasets)
function class_weights = calculate_class_weights(y, method)
    %CALCULATE_CLASS_WEIGHTS Calculate weights for imbalanced classes
    %
    % Inputs:
    %   y - Labels
    %   method - 'balanced' or 'frequency'
    %
    % Outputs:
    %   class_weights - Weight for each class
    
    if nargin < 2
        method = 'balanced';
    end
    
    % Get unique classes and their counts
    data_cat = categorical(y);
    [counts, classes] = hist(data_cat, unique(data_cat));
    n_classes = length(classes);
    n_samples = length(y);
    
    switch method
        case 'balanced'
            % Balanced: weight = total_samples / (n_classes * class_count)
            class_weights = n_samples ./ (n_classes * counts);
        case 'frequency'
            % Frequency: weight = total_samples / class_count
            class_weights = n_samples ./ counts;
        otherwise
            class_weights = ones(n_classes, 1);
    end
    
    % Normalize weights
    class_weights = class_weights / sum(class_weights) * n_classes;
end

%% Data Augmentation Function
function X_aug = augment_signals(X, augmentation_factor)
    %AUGMENT_SIGNALS Apply data augmentation to ECG signals
    %
    % Simple augmentation techniques:
    % 1. Time shifting
    % 2. Amplitude scaling
    % 3. Noise injection
    %
    % Inputs:
    %   X - Input signals [leads x samples x segments]
    %   augmentation_factor - How many times to augment the data
    %
    % Outputs:
    %   X_aug - Augmented signals
    
    [n_leads, n_samples, n_segments] = size(X);
    
    % Calculate number of augmentations per segment
    n_aug_per_segment = max(1, floor(augmentation_factor - 1));
    
    % Initialize augmented data
    X_aug = zeros(n_leads, n_samples, n_segments * (n_aug_per_segment + 1), 'single');
    
    % Original data
    X_aug(:, :, 1:n_segments) = X;
    
    % Generate augmentations
    aug_idx = n_segments + 1;
    for i = 1:n_segments
        segment = X(:, :, i);
        
        for j = 1:n_aug_per_segment
            % Random choice of augmentation
            aug_type = randi([1, 3]);
            
            switch aug_type
                case 1  % Time shift
                    shift = randi([-floor(n_samples*0.1), floor(n_samples*0.1)]);
                    aug_segment = circshift(segment, shift, 2);
                    
                case 2  % Amplitude scaling
                    scale = 0.9 + 0.2 * rand();  % 0.9-1.1x scaling
                    aug_segment = segment * scale;
                    
                case 3  % Gaussian noise
                    noise = 0.01 * randn(size(segment));
                    aug_segment = segment + noise;
            end
            
            X_aug(:, :, aug_idx) = aug_segment;
            aug_idx = aug_idx + 1;
        end
    end
end
