classdef DataLoader
    % DataLoader - Handles loading, preprocessing, and batching of ECG data from PTB DB
    
    properties
        ptb_path
        sampling_rate
        signal_length
        num_leads
        data_dir
    end
    
    methods
        function obj = DataLoader(config)
            % Initialize DataLoader with project configuration
            obj.ptb_path = config.PTB_DB_PATH;
            obj.sampling_rate = config.SAMPLING_RATE;
            obj.signal_length = config.SIGNAL_LENGTH;
            obj.num_leads = config.NUM_LEADS;
            obj.data_dir = config.DATA_DIR;
        end
        
        function [signal, fs, time] = loadRawSignal(obj, patientFolder, recordName)
            % Load raw ECG signal from PTB DB using rdsamp
            % Returns: signal (samples x leads), fs (sampling rate), time (samples x 1)
            
            try
                oldFolder = pwd;
                cd(patientFolder);
                [signal, fs, time] = rdsamp(recordName);
                cd(oldFolder);
                
                fprintf('[DataLoader] Loaded: %s - Shape: (%d, %d)\n', ...
                    recordName, size(signal,1), size(signal,2));
            catch ME
                cd(oldFolder);
                error('[DataLoader] Failed to load %s: %s', recordName, ME.message);
            end
        end
        
        function signal_normalized = normalizeSignal(obj, signal)
            % Normalize each lead independently (z-score normalization)
            % Input: signal (samples x leads)
            % Output: normalized signal (samples x leads)
            
            signal_normalized = zeros(size(signal));
            
            for lead = 1:size(signal, 2)
                lead_data = signal(:, lead);
                mean_val = mean(lead_data);
                std_val = std(lead_data);
                
                if std_val > 0
                    signal_normalized(:, lead) = (lead_data - mean_val) / std_val;
                else
                    signal_normalized(:, lead) = lead_data - mean_val;
                end
            end
        end
        
        function signal_reshaped = reshapeSignal(obj, signal, target_length)
            % Reshape signal to target length
            % If signal is longer: truncate from the middle (keep steady-state)
            % If signal is shorter: pad with zeros
            
            current_length = size(signal, 1);
            
            if current_length == target_length
                signal_reshaped = signal;
            elseif current_length > target_length
                % Truncate: remove beginning and end, keep middle
                start_idx = floor((current_length - target_length) / 2) + 1;
                signal_reshaped = signal(start_idx:start_idx + target_length - 1, :);
            else
                % Pad with zeros
                padding = target_length - current_length;
                pad_before = floor(padding / 2);
                pad_after = padding - pad_before;
                signal_reshaped = [zeros(pad_before, size(signal,2)); 
                                   signal; 
                                   zeros(pad_after, size(signal,2))];
            end
        end
        
        function signal_processed = preprocessSignal(obj, signal, varargin)
            % Complete preprocessing pipeline
            % Options: 'normalize', 'reshape', 'filter'
            
            signal_processed = signal;
            
            % Normalize
            if ismember('normalize', varargin)
                signal_processed = obj.normalizeSignal(signal_processed);
            end
            
            % Reshape to standard length
            if ismember('reshape', varargin)
                signal_processed = obj.reshapeSignal(signal_processed, obj.signal_length);
            end
            
            % Filter (optional)
            if ismember('filter', varargin)
                signal_processed = obj.filterSignal(signal_processed);
            end
        end
        
        function signal_filtered = filterSignal(obj, signal)
            % Apply bandpass filter (0.5-40 Hz for ECG)
            % Input: signal (samples x leads)
            
            Fs = obj.sampling_rate;
            Fpass = [0.5 40];  % Passband
            Fstop = [0.1 50];  % Stopband
            
            % Design filter
            d = designfilt('bandpass', ...
                'StopbandFrequency1', Fstop(1), ...
                'PassbandFrequency1', Fpass(1), ...
                'PassbandFrequency2', Fpass(2), ...
                'StopbandFrequency2', Fstop(2), ...
                'StopbandAttenuation1', 40, ...
                'PassbandRipple', 1, ...
                'StopbandAttenuation2', 40, ...
                'SampleRate', Fs);
            
            signal_filtered = zeros(size(signal));
            for lead = 1:size(signal, 2)
                signal_filtered(:, lead) = filtfilt(d, signal(:, lead));
            end
        end
        
        function dataset = loadPatientData(obj, patientNum, recordName)
            % Load and preprocess data from a single patient
            % Returns: dataset struct with processed signal and metadata
            
            patientFolder = fullfile(obj.ptb_path, ...
                sprintf('patient%03d', patientNum));
            
            if ~isfolder(patientFolder)
                error('[DataLoader] Patient folder not found: %s', patientFolder);
            end
            
            % Load raw signal
            [signal, fs, time] = obj.loadRawSignal(patientFolder, recordName);
            
            % Preprocess
            signal_processed = obj.preprocessSignal(signal, 'normalize', 'reshape', 'filter');
            
            % Create dataset struct
            dataset = struct();
            dataset.patient_id = patientNum;
            dataset.record_name = recordName;
            dataset.signal = signal_processed;
            dataset.fs = fs;
            dataset.time = time;
            dataset.num_leads = size(signal_processed, 2);
            dataset.signal_length = size(signal_processed, 1);
        end
        
        function [data_train, data_val, labels_train, labels_val] = ...
            createTrainValSplit(obj, data, labels, train_ratio)
            % Create train/validation split
            % Input: data (samples x leads x patients), labels (patients x 1)
            
            n_samples = length(labels);
            n_train = floor(n_samples * train_ratio);
            
            % Random permutation
            idx = randperm(n_samples);
            idx_train = idx(1:n_train);
            idx_val = idx(n_train+1:end);
            
            % Split data
            data_train = data(:, :, idx_train);
            data_val = data(:, :, idx_val);
            labels_train = labels(idx_train);
            labels_val = labels(idx_val);
            
            fprintf('[DataLoader] Train: %d, Validation: %d\n', ...
                length(labels_train), length(labels_val));
        end
        
        function [batch_data, batch_labels] = getMiniBatch(obj, data, labels, batch_idx, batch_size)
            % Get a mini-batch for training
            % Input: batch_idx (batch number), batch_size
            
            start_idx = (batch_idx - 1) * batch_size + 1;
            end_idx = min(batch_idx * batch_size, size(data, 3));
            
            batch_data = data(:, :, start_idx:end_idx);
            batch_labels = labels(start_idx:end_idx);
        end
    end
end
