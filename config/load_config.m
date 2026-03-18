%% Configuration Function for ECG CNN Classification System
function config = load_config()
    %LOAD_CONFIG Load configuration parameters for the ECG classification system
    %
    % Returns:
    %   config - Structure containing all configuration parameters
    
    % Data paths
    config.ptb_path = '/home/subhajitroy005/Documents/Projects/ECG/PTB_DB';
    config.matlab_path = '/home/subhajitroy005/Documents/Projects/ECG/matlab/mcode';
    config.output_path = './results';
    
    % Data parameters
    config.max_patients = 100;           % Max patients to load (-1 for all)
    config.sampling_freq = 250;          % Hz (PTB standard)
    config.signal_duration = 10;         % seconds
    config.signal_length = config.sampling_freq * config.signal_duration; % samples
    
    % Data preprocessing
    config.normalize_method = 'zscore';  % 'zscore', 'minmax', or 'robust'
    config.filter_type = 'butterworth'; % 'butterworth' or 'none'
    config.filter_order = 4;
    config.lowpass_freq = 40;            % Hz
    config.highpass_freq = 0.5;          % Hz
    config.remove_outliers = true;
    config.outlier_threshold = 3;        % std deviations
    
    % Segmentation parameters
    config.segment_length = 512;         % Samples per segment (ECG window)
    config.segment_overlap = 0.5;        % 50% overlap
    config.augment_data = false;         % Data augmentation
    config.augmentation_factor = 1.5;    % Create 1.5x more segments
    
    % CNN Architecture parameters
    config.input_length = config.segment_length;
    config.num_leads = 12;               % PTB has 12-lead ECG
    config.num_classes = 5;              % Adjust based on your classification task
                                         % 1: Normal, 2: MI, 3: CLBBB, 4: CRBBB, 5: Others
    
    % CNN Architecture
    config.cnn_architecture = struct();
    config.cnn_architecture.num_filters_layer1 = 32;
    config.cnn_architecture.num_filters_layer2 = 64;
    config.cnn_architecture.num_filters_layer3 = 128;
    config.cnn_architecture.filter_size = 5;
    config.cnn_architecture.pool_size = 2;
    config.cnn_architecture.dropout_rate = 0.3;
    config.cnn_architecture.fc_units = 128;
    
    % Training parameters
    config.random_seed = 42;
    config.train_test_ratio = 0.8;       % 80% train, 20% test
    config.validation_split = 0.1;       % 10% of training for validation
    config.batch_size = 32;
    config.max_epochs = 100;
    config.learning_rate = 0.001;
    config.optimizer = 'adam';           % 'adam', 'sgd', 'rmsprop'
    config.weight_decay = 1e-5;
    
    % Early stopping and checkpointing
    config.early_stopping = true;
    config.early_stopping_patience = 15; % epochs
    config.save_best_model = true;
    
    % Evaluation parameters
    config.compute_roc = true;
    config.compute_precision_recall = true;
    config.k_fold_cv = false;            % 5-fold cross-validation
    config.k_folds = 5;
    
    % Class labels (adjust based on your classification task)
    config.class_names = categorical([
        "Normal"
        "MI"
        "CLBBB"
        "CRBBB"
        "Other"
    ]);
    
    % System parameters
    config.use_gpu = true;
    config.verbose = true;
    config.random_seed_numpy = 42;
    
    % File format and extensions
    config.ecg_extension = '.mat';       % PTB format
    config.save_format = 'mat';          % 'mat' or 'h5'
    
    % Logging
    config.log_file = './results/training_log.txt';
    config.plot_figures = true;
    config.save_figures = true;
    config.figure_format = 'png';        % 'png', 'jpg', 'fig'
    config.figure_dpi = 300;
    
end
