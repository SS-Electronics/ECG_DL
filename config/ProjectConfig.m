classdef ProjectConfig
    % ProjectConfig - Central configuration for ECG CNN classification project
    % Manages paths, data sources, model settings, and training parameters
    
    properties (Constant)
        % Project paths
        PROJECT_ROOT = pwd;
        DATA_DIR = fullfile(pwd, 'data');
        PTB_DB_PATH = '/home/subhajitroy005/Documents/Projects/ECG/PTB_DB';
        MODELS_DIR = fullfile(pwd, 'models');
        RESULTS_DIR = fullfile(pwd, 'results');
        LOGS_DIR = fullfile(pwd, 'logs');
        
        % MATLAB path setup
        MATLAB_MCODE_PATH = '/home/subhajitroy005/Documents/Projects/ECG/matlab/mcode';
        
        % Database configuration
        PTBDB_PATIENTS = 549;  % Total patients in PTB DB
        SAMPLING_RATE = 1000;   % Hz (PTB DB standard)
        SIGNAL_DURATION = 10;   % seconds
        SIGNAL_LENGTH = 10000;  % samples (Fs * duration)
        NUM_LEADS = 12;         % 12-lead ECG
        
        % Training configuration
        TRAIN_VAL_SPLIT = 0.8;      % 80% train, 20% validation
        BATCH_SIZE = 32;
        NUM_EPOCHS = 100;
        LEARNING_RATE = 0.001;
        VALIDATION_PATIENCE = 15;   % Early stopping patience
        
        % Data augmentation
        AUGMENTATION_ENABLED = true;
        AUGMENTATION_FACTOR = 2;    % Generate 2x more augmented samples
        
        % Disease classification labels
        DISEASE_LABELS = {
            'Normal'
            'MI (Myocardial Infarction)'
            'LBBB (Left Bundle Branch Block)'
            'RBBB (Right Bundle Branch Block)'
            'SB (Sinus Bradycardia)'
            'AF (Atrial Fibrillation)'
        };
        
        % Model hyperparameters (can be overridden per model)
        DEFAULT_MODEL_PARAMS = struct(...
            'input_size', 10000, ...
            'num_leads', 12, ...
            'num_classes', 6, ...
            'dropout_rate', 0.5, ...
            'l2_regularization', 1e-4);
    end
    
    methods (Static)
        function initialize()
            % Initialize project structure and MATLAB paths
            ProjectConfig.setupPaths();
            ProjectConfig.createDirectories();
            ProjectConfig.validateConfiguration();
        end
        
        function setupPaths()
            % Add required paths to MATLAB
            addpath(ProjectConfig.MATLAB_MCODE_PATH);
            addpath(fullfile(ProjectConfig.PROJECT_ROOT, 'core'));
            addpath(fullfile(ProjectConfig.PROJECT_ROOT, 'data'));
            addpath(fullfile(ProjectConfig.PROJECT_ROOT, 'models'));
            addpath(fullfile(ProjectConfig.PROJECT_ROOT, 'training'));
            addpath(fullfile(ProjectConfig.PROJECT_ROOT, 'utils'));
            addpath(fullfile(ProjectConfig.PROJECT_ROOT, 'config'));
            addpath(fullfile(ProjectConfig.PROJECT_ROOT, 'visualization'));
        end
        
        function createDirectories()
            % Create necessary project directories
            dirs = {
                ProjectConfig.DATA_DIR
                ProjectConfig.MODELS_DIR
                ProjectConfig.RESULTS_DIR
                ProjectConfig.LOGS_DIR
                fullfile(ProjectConfig.DATA_DIR, 'raw')
                fullfile(ProjectConfig.DATA_DIR, 'processed')
                fullfile(ProjectConfig.DATA_DIR, 'augmented')
                fullfile(ProjectConfig.MODELS_DIR, 'trained')
                fullfile(ProjectConfig.MODELS_DIR, 'weights')
                fullfile(ProjectConfig.RESULTS_DIR, 'plots')
                fullfile(ProjectConfig.RESULTS_DIR, 'metrics')
            };
            
            for i = 1:length(dirs)
                if ~isfolder(dirs{i})
                    mkdir(dirs{i});
                    fprintf('[Config] Created directory: %s\n', dirs{i});
                end
            end
        end
        
        function validateConfiguration()
            % Validate that all paths and dependencies exist
            if ~isfolder(ProjectConfig.PTB_DB_PATH)
                warning('[Config] PTB DB path not found: %s', ProjectConfig.PTB_DB_PATH);
            end
            
            if ~exist(ProjectConfig.MATLAB_MCODE_PATH, 'dir')
                warning('[Config] MATLAB mcode path not found: %s', ProjectConfig.MATLAB_MCODE_PATH);
            end
            
            fprintf('[Config] Project initialized successfully\n');
        end
        
        function printConfig()
            % Print current configuration
            fprintf('\n========== PROJECT CONFIGURATION ==========\n');
            fprintf('Project Root: %s\n', ProjectConfig.PROJECT_ROOT);
            fprintf('PTB DB Path: %s\n', ProjectConfig.PTB_DB_PATH);
            fprintf('Sampling Rate: %d Hz\n', ProjectConfig.SAMPLING_RATE);
            fprintf('Signal Length: %d samples\n', ProjectConfig.SIGNAL_LENGTH);
            fprintf('Number of Leads: %d\n', ProjectConfig.NUM_LEADS);
            fprintf('Training/Validation Split: %.1f%%\n', ProjectConfig.TRAIN_VAL_SPLIT * 100);
            fprintf('Batch Size: %d\n', ProjectConfig.BATCH_SIZE);
            fprintf('Number of Epochs: %d\n', ProjectConfig.NUM_EPOCHS);
            fprintf('Learning Rate: %.4f\n', ProjectConfig.LEARNING_RATE);
            fprintf('Augmentation Enabled: %s\n', ProjectConfig.AUGMENTATION_ENABLED);
            fprintf('Disease Labels: %d\n', length(ProjectConfig.DISEASE_LABELS));
            fprintf('==========================================\n\n');
        end
    end
end
