classdef PTBDataLoader < handle
    % PTBDataLoader - Load and preprocess real ECG data from PTB Database
    % Handles patient data, labels, and disease classification
    
    properties
        ptb_path                % Path to PTB Database
        sampling_rate           % Hz
        signal_length           % Target length
        num_leads              % Number of ECG leads
        patient_data           % Cache for loaded patient data
        disease_labels         % Cache for disease labels
    end
    
    properties (Constant)
        % Disease diagnosis codes
        DISEASE_CODES = struct(...
            'Normal', 0, ...
            'MI', 1, ...
            'LBBB', 2, ...
            'RBBB', 3, ...
            'SB', 4, ...
            'AF', 5);
    end
    
    methods
        function obj = PTBDataLoader(ptb_path, sampling_rate, signal_length, num_leads)
            % Initialize PTB data loader
            obj.ptb_path = ptb_path;
            obj.sampling_rate = sampling_rate;
            obj.signal_length = signal_length;
            obj.num_leads = num_leads;
            obj.patient_data = struct();
            obj.disease_labels = struct();
            
            fprintf('[PTBDataLoader] Initialized for path: %s\n', ptb_path);
        end
        
        function [signal, fs, time] = loadPatientRecord(obj, patient_num, record_name)
            % Load individual patient ECG record using rdsamp
            % Returns: signal (samples x leads), fs (sampling rate), time
            
            patient_dir = fullfile(obj.ptb_path, sprintf('patient%03d', patient_num));
            
            if ~isfolder(patient_dir)
                error('[PTBDataLoader] Patient directory not found: %s', patient_dir);
            end
            
            try
                % Save current directory
                old_dir = pwd;
                
                % Change to patient directory
                cd(patient_dir);
                
                % Load using rdsamp (requires MATLAB ECG/biosignal tools)
                [signal, fs, time] = rdsamp(record_name);
                
                % Return to original directory
                cd(old_dir);
                
                fprintf('[PTBDataLoader] Loaded patient %03d, record %s: (%d, %d)\n', ...
                    patient_num, record_name, size(signal, 1), size(signal, 2));
                
            catch ME
                cd(old_dir);
                error('[PTBDataLoader] Failed to load record: %s', ME.message);
            end
        end
        
        function signal_normalized = normalizeSignal(obj, signal)
            % Normalize each lead independently (z-score normalization)
            % Removes baseline wander and normalizes amplitude
            
            signal_normalized = zeros(size(signal));
            
            for lead = 1:size(signal, 2)
                lead_data = signal(:, lead);
                
                % Remove DC offset
                lead_data = lead_data - mean(lead_data);
                
                % Normalize by standard deviation
                std_val = std(lead_data);
                if std_val > 0
                    signal_normalized(:, lead) = lead_data / std_val;
                else
                    signal_normalized(:, lead) = lead_data;
                end
            end
            
            fprintf('[PTBDataLoader] Signal normalized (z-score per lead)\n');
        end
        
        function signal_filtered = filterSignal(obj, signal)
            % Apply bandpass filter to remove noise
            % ECG typical bandwidth: 0.5 Hz - 40 Hz
            
            Fs = obj.sampling_rate;
            
            % Design bandpass filter
            Fpass = [0.5 40];      % Passband frequencies
            Fstop = [0.1 50];      % Stopband frequencies
            
            try
                d = designfilt('bandpass', ...
                    'StopbandFrequency1', Fstop(1), ...
                    'PassbandFrequency1', Fpass(1), ...
                    'PassbandFrequency2', Fpass(2), ...
                    'StopbandFrequency2', Fstop(2), ...
                    'StopbandAttenuation1', 40, ...
                    'PassbandRipple', 1, ...
                    'StopbandAttenuation2', 40, ...
                    'SampleRate', Fs);
                
                % Apply filter to each lead
                signal_filtered = zeros(size(signal));
                for lead = 1:size(signal, 2)
                    signal_filtered(:, lead) = filtfilt(d, signal(:, lead));
                end
                
                fprintf('[PTBDataLoader] Bandpass filter applied (0.5-40 Hz)\n');
                
            catch ME
                warning('[PTBDataLoader] Filtering failed: %s', ME.message);
                warning('[PTBDataLoader] Using raw signal (install Signal Processing Toolbox)\n');
                signal_filtered = signal;
            end
        end
        
        function signal_reshaped = reshapeSignal(obj, signal, target_length)
            % Reshape signal to target length
            % If longer: truncate from middle (keep steady-state)
            % If shorter: zero-pad symmetrically
            
            current_length = size(signal, 1);
            
            if current_length == target_length
                signal_reshaped = signal;
                
            elseif current_length > target_length
                % Truncate: remove beginning and end, keep middle steady-state
                trim_total = current_length - target_length;
                trim_start = floor(trim_total / 2);
                trim_end = trim_start + target_length - 1;
                
                signal_reshaped = signal(trim_start:trim_end, :);
                fprintf('[PTBDataLoader] Truncated: %d -> %d samples\n', current_length, target_length);
                
            else
                % Pad with zeros symmetrically
                padding_total = target_length - current_length;
                pad_start = floor(padding_total / 2);
                pad_end = padding_total - pad_start;
                
                signal_reshaped = [
                    zeros(pad_start, size(signal, 2));
                    signal;
                    zeros(pad_end, size(signal, 2))
                ];
                fprintf('[PTBDataLoader] Zero-padded: %d -> %d samples\n', current_length, target_length);
            end
        end
        
        function signal_resampled = resampleSignal(obj, signal, original_fs, target_fs)
            % Resample signal to different sampling rate if needed
            
            if original_fs == target_fs
                signal_resampled = signal;
                return;
            end
            
            ratio = target_fs / original_fs;
            new_length = round(size(signal, 1) * ratio);
            
            try
                signal_resampled = zeros(new_length, size(signal, 2));
                for lead = 1:size(signal, 2)
                    signal_resampled(:, lead) = resample(signal(:, lead), target_fs, original_fs);
                end
                fprintf('[PTBDataLoader] Resampled: %.0f Hz -> %.0f Hz\n', original_fs, target_fs);
            catch
                warning('[PTBDataLoader] Resampling failed, using interpolation\n');
                old_t = linspace(0, 1, size(signal, 1));
                new_t = linspace(0, 1, new_length);
                signal_resampled = interp1(old_t, signal, new_t);
            end
        end
        
        function signal_denoised = denoiseSignal(obj, signal, wavelet_level)
            % Denoise using wavelet decomposition
            % Removes high-frequency noise while preserving ECG features
            
            if nargin < 3
                wavelet_level = 4;
            end
            
            signal_denoised = zeros(size(signal));
            
            try
                for lead = 1:size(signal, 2)
                    % Wavelet decomposition
                    [C, L] = wavedec(signal(:, lead), wavelet_level, 'db4');
                    
                    % Estimate noise level (Donoho method)
                    sigma = median(abs(C(L(1)+1:L(1)+L(2)))) / 0.6745;
                    
                    % Compute threshold
                    threshold = sigma * sqrt(2 * log(length(signal)));
                    
                    % Soft thresholding
                    C_thresh = wthresh(C, 's', threshold);
                    
                    % Reconstruct
                    signal_denoised(:, lead) = waverec(C_thresh, L, 'db4');
                    signal_denoised(1:size(signal, 1), lead) = signal_denoised(1:size(signal, 1), lead);
                end
                
                fprintf('[PTBDataLoader] Wavelet denoising applied (db4, level %d)\n', wavelet_level);
                
            catch ME
                warning('[PTBDataLoader] Wavelet denoising failed: %s\n', ME.message);
                signal_denoised = signal;
            end
        end
        
        function signal_processed = preprocessSignal(obj, signal, preprocessing_config)
            % Complete preprocessing pipeline
            % preprocessing_config: struct with fields:
            %   - normalize: true/false
            %   - filter: true/false
            %   - denoise: true/false
            %   - reshape: target_length or empty
            
            signal_processed = signal;
            
            % Default configuration
            if nargin < 3
                preprocessing_config = struct(...
                    'normalize', true, ...
                    'filter', true, ...
                    'denoise', false, ...
                    'reshape', obj.signal_length);
            end
            
            fprintf('[PTBDataLoader] Starting preprocessing pipeline:\n');
            
            % Normalization
            if preprocessing_config.normalize
                signal_processed = obj.normalizeSignal(signal_processed);
            end
            
            % Filtering
            if preprocessing_config.filter
                signal_processed = obj.filterSignal(signal_processed);
            end
            
            % Denoising (optional, slower)
            if preprocessing_config.denoise
                signal_processed = obj.denoiseSignal(signal_processed);
            end
            
            % Reshape
            if ~isempty(preprocessing_config.reshape)
                signal_processed = obj.reshapeSignal(signal_processed, preprocessing_config.reshape);
            end
            
            fprintf('[PTBDataLoader] Preprocessing complete\n');
        end
        
        function [dataset] = loadPatientDataset(obj, patient_num, record_names, disease_label)
            % Load complete patient dataset with preprocessing
            % dataset: struct with signal, labels, metadata
            
            fprintf('[PTBDataLoader] Loading patient %03d dataset...\n', patient_num);
            
            dataset = struct();
            dataset.patient_id = patient_num;
            dataset.disease_label = disease_label;
            dataset.records = {};
            dataset.signals = [];
            dataset.fs = obj.sampling_rate;
            
            n_records = length(record_names);
            
            for rec_idx = 1:n_records
                try
                    % Load raw signal
                    record_name = record_names{rec_idx};
                    [signal, fs, time] = obj.loadPatientRecord(patient_num, record_name);
                    
                    % Store original sampling rate
                    if fs ~= obj.sampling_rate
                        fprintf('[PTBDataLoader] Resampling from %.0f to %.0f Hz\n', fs, obj.sampling_rate);
                        signal = obj.resampleSignal(signal, fs, obj.sampling_rate);
                    end
                    
                    % Preprocess
                    config = struct('normalize', true, 'filter', true, ...
                        'denoise', false, 'reshape', obj.signal_length);
                    signal_processed = obj.preprocessSignal(signal, config);
                    
                    % Ensure correct shape
                    if size(signal_processed, 1) ~= obj.signal_length
                        signal_processed = obj.reshapeSignal(signal_processed, obj.signal_length);
                    end
                    
                    if size(signal_processed, 2) ~= obj.num_leads
                        % Select only first num_leads if more available
                        if size(signal_processed, 2) > obj.num_leads
                            signal_processed = signal_processed(:, 1:obj.num_leads);
                        else
                            % Pad with zeros if fewer leads
                            padding = zeros(size(signal_processed, 1), obj.num_leads - size(signal_processed, 2));
                            signal_processed = [signal_processed, padding];
                        end
                    end
                    
                    % Store
                    dataset.records{rec_idx} = record_name;
                    dataset.signals = cat(3, dataset.signals, signal_processed);
                    
                catch ME
                    fprintf('[PTBDataLoader] ⚠ Record %s failed: %s\n', record_name, ME.message);
                end
            end
            
            dataset.num_records = length(dataset.records);
            fprintf('[PTBDataLoader] Loaded %d records for patient %03d\n', ...
                dataset.num_records, patient_num);
        end
        
        function [data, labels] = loadPatientBatch(obj, patient_nums, record_names_cell, disease_labels)
            % Load batch of patients
            % patient_nums: array of patient IDs
            % record_names_cell: cell array of record names per patient
            % disease_labels: array of disease class labels
            
            n_patients = length(patient_nums);
            data = [];
            labels = [];
            
            fprintf('[PTBDataLoader] Loading batch of %d patients...\n', n_patients);
            
            for p_idx = 1:n_patients
                patient_id = patient_nums(p_idx);
                record_names = record_names_cell{p_idx};
                disease_label = disease_labels(p_idx);
                
                try
                    % Load patient dataset
                    dataset = obj.loadPatientDataset(patient_id, record_names, disease_label);
                    
                    % Extract signals (average across multiple records if available)
                    if dataset.num_records > 0
                        patient_signal = mean(dataset.signals, 3);  % Average across records
                        
                        % Stack with previous data
                        data = cat(3, data, patient_signal);
                        
                        % Store label (one-hot encoded)
                        one_hot = zeros(1, 6);
                        one_hot(disease_label) = 1;
                        labels = [labels; one_hot];
                        
                        fprintf('[PTBDataLoader] ✓ Patient %03d loaded (class %d)\n', ...
                            patient_id, disease_label);
                    end
                    
                catch ME
                    fprintf('[PTBDataLoader] ✗ Patient %03d failed: %s\n', patient_id, ME.message);
                end
            end
            
            fprintf('[PTBDataLoader] Batch loading complete: %d patients loaded\n', ...
                size(data, 3));
        end
        
        function [train_data, val_data, test_data, train_labels, val_labels, test_labels] = ...
            createDataSplit(obj, data, labels, train_ratio, val_ratio)
            % Create train/validation/test split
            % data: (samples x leads x n_patients)
            % labels: (n_patients x 6) one-hot encoded
            % train_ratio: 0.6 (60% train)
            % val_ratio: 0.2 (20% val, 20% test)
            
            if nargin < 4
                train_ratio = 0.6;
            end
            if nargin < 5
                val_ratio = 0.2;
            end
            
            test_ratio = 1 - train_ratio - val_ratio;
            
            n_samples = size(data, 3);
            
            % Random permutation
            idx = randperm(n_samples);
            
            % Calculate split points
            n_train = round(n_samples * train_ratio);
            n_val = round(n_samples * val_ratio);
            
            idx_train = idx(1:n_train);
            idx_val = idx(n_train+1:n_train+n_val);
            idx_test = idx(n_train+n_val+1:end);
            
            % Split data
            train_data = data(:, :, idx_train);
            val_data = data(:, :, idx_val);
            test_data = data(:, :, idx_test);
            
            % Split labels
            train_labels = labels(idx_train, :);
            val_labels = labels(idx_val, :);
            test_labels = labels(idx_test, :);
            
            fprintf('[PTBDataLoader] Data split:\n');
            fprintf('  Train: %d samples (%.1f%%)\n', size(train_data, 3), train_ratio*100);
            fprintf('  Val:   %d samples (%.1f%%)\n', size(val_data, 3), val_ratio*100);
            fprintf('  Test:  %d samples (%.1f%%)\n', size(test_data, 3), test_ratio*100);
        end
        
        function augmented_data = augmentData(obj, data, augmentation_config)
            % Data augmentation: generate synthetic variations
            % Techniques: scaling, noise, time stretching, shifting
            
            if nargin < 3
                augmentation_config = struct(...
                    'scale_factor', [0.95 1.05], ...
                    'noise_std', 0.01, ...
                    'time_shift', true, ...
                    'n_augments', 2);
            end
            
            n_originals = size(data, 3);
            n_augments = augmentation_config.n_augments;
            
            augmented_data = data;  % Start with originals
            
            fprintf('[PTBDataLoader] Augmenting data (%d× multiplication)...\n', n_augments);
            
            for aug = 1:n_augments
                for orig = 1:n_originals
                    signal = data(:, :, orig);
                    
                    % Random scaling
                    scale = augmentation_config.scale_factor(1) + ...
                        rand() * (augmentation_config.scale_factor(2) - augmentation_config.scale_factor(1));
                    signal = signal * scale;
                    
                    % Add Gaussian noise
                    noise = randn(size(signal)) * augmentation_config.noise_std;
                    signal = signal + noise;
                    
                    % Time shifting (circular shift)
                    if augmentation_config.time_shift
                        shift = randi([-100, 100]);  % Shift by ±100 samples
                        signal = circshift(signal, shift, 1);
                    end
                    
                    % Append augmented sample
                    augmented_data = cat(3, augmented_data, signal);
                end
            end
            
            fprintf('[PTBDataLoader] Augmentation complete: %d -> %d samples\n', ...
                n_originals, size(augmented_data, 3));
        end
        
        function printStatistics(obj, data, labels, dataset_name)
            % Print dataset statistics
            
            if nargin < 4
                dataset_name = 'Dataset';
            end
            
            fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
            fprintf('║  %s Statistics\n', dataset_name);
            fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
            
            fprintf('Data Shape: %d × %d × %d\n', size(data, 1), size(data, 2), size(data, 3));
            fprintf('Labels Shape: %d × %d\n', size(labels, 1), size(labels, 2));
            
            fprintf('\nSamples per Class:\n');
            class_counts = sum(labels, 1);
            for c = 1:length(class_counts)
                fprintf('  Class %d: %d samples\n', c, class_counts(c));
            end
            
            fprintf('\nSignal Statistics:\n');
            fprintf('  Min: %.4f\n', min(data(:)));
            fprintf('  Max: %.4f\n', max(data(:)));
            fprintf('  Mean: %.4f\n', mean(data(:)));
            fprintf('  Std: %.4f\n', std(data(:)));
        end
    end
end
