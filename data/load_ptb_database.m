%% Load PTB Database Function
function [signals, labels] = load_ptb_database(ptb_path, max_patients)
    %LOAD_PTB_DATABASE Load ECG signals from PTB database
    %
    % Inputs:
    %   ptb_path - Path to PTB database root directory
    %   max_patients - Maximum number of patients to load (-1 for all)
    %
    % Outputs:
    %   signals - Cell array of ECG signals [samples x leads]
    %   labels - Array of class labels for each signal
    
    addpath('/home/subhajitroy005/Documents/Projects/ECG/matlab/mcode');
    
    signals = {};
    labels = [];
    signal_count = 0;
    
    % List all patient folders
    patient_dirs = dir(fullfile(ptb_path, 'patient*'));
    
    if isempty(patient_dirs)
        error('No patient folders found in %s', ptb_path);
    end
    
    % Limit number of patients if specified
    if max_patients > 0
        patient_dirs = patient_dirs(1:min(max_patients, length(patient_dirs)));
    end
    
    fprintf('Found %d patient folders\n', length(patient_dirs));
    
    % Load signals from each patient
    for p = 1:length(patient_dirs)
        patient_folder = fullfile(ptb_path, patient_dirs(p).name);
        
        % List all ECG records in patient folder
        record_files = dir(fullfile(patient_folder, '*.hea'));
        
        if isempty(record_files)
            continue;
        end
        
        % Load each record
        for r = 1:length(record_files)
            % Get record name without extension
            record_name = record_files(r).name(1:end-4);
            
            try
                % Read signal using rdsamp
                [signal, fs, tm] = rdsamp(fullfile(patient_folder, record_name));
                
                % Basic validation
                if size(signal, 1) < 500  % Minimum signal length
                    continue;
                end
                
                % Store signal
                signal_count = signal_count + 1;
                signals{signal_count} = signal;
                
                % Assign label based on patient ID and diagnosis
                % This is a placeholder - adjust based on your classification task
                label = assign_ecg_label(patient_dirs(p).name, record_name);
                labels = [labels; label];
                
                if mod(signal_count, 50) == 0
                    fprintf('  Loaded %d signals...\n', signal_count);
                end
                
            catch ME
                % Skip files that can't be read
                if ~strcmp(ME.identifier, 'MATLAB:toc:invalidTimerHandle')
                    fprintf('Warning: Could not load %s/%s: %s\n', ...
                        patient_dirs(p).name, record_name, ME.message);
                end
            end
        end
        
        fprintf('Processed patient %d/%d (%s): %d signals loaded\n', ...
            p, length(patient_dirs), patient_dirs(p).name, signal_count);
    end
    
    fprintf('\nTotal signals loaded: %d\n', signal_count);
    
    if signal_count == 0
        error('No ECG signals were successfully loaded');
    end
end

%% Assign Label Function
function label = assign_ecg_label(patient_id, record_name)
    %ASSIGN_ECG_LABEL Assign diagnostic label to ECG signal
    %
    % This is a placeholder function. You should modify it based on:
    % 1. PTB database annotations
    % 2. Your classification task
    % 3. Available diagnostic information
    %
    % Common PTB diagnoses:
    % 1: Normal ECG
    % 2: Myocardial Infarction (MI)
    % 3: Left Bundle Branch Block (CLBBB)
    % 4: Right Bundle Branch Block (CRBBB)
    % 5: Other abnormalities
    
    % Extract patient number from ID (e.g., 'patient001' -> 1)
    patient_num = str2double(patient_id(8:end));
    
    % Simple labeling strategy (replace with actual PTB labels)
    if patient_num <= 20
        label = 1;  % Normal
    elseif patient_num <= 40
        label = 2;  % MI
    elseif patient_num <= 60
        label = 3;  % CLBBB
    elseif patient_num <= 80
        label = 4;  % CRBBB
    else
        label = 5;  % Other
    end
end

%% Preprocess Signals Function
function [processed_signals, info] = preprocess_signals(signals, config)
    %PREPROCESS_SIGNALS Apply preprocessing to ECG signals
    %
    % Inputs:
    %   signals - Cell array of raw ECG signals
    %   config - Configuration structure
    %
    % Outputs:
    %   processed_signals - Preprocessed signals
    %   info - Preprocessing information
    
    processed_signals = cell(size(signals));
    info = struct();
    
    fprintf('Preprocessing %d signals...\n', length(signals));
    
    for i = 1:length(signals)
        signal = signals{i};
        
        % 1. Normalize signal
        switch config.normalize_method
            case 'zscore'
                signal = zscore(signal, 0, 1);
            case 'minmax'
                signal = normalize(signal, 'range');
            case 'robust'
                signal = (signal - median(signal, 1)) / iqr(signal, 1);
        end
        
        % 2. Remove outliers
        if config.remove_outliers
            outlier_mask = abs(signal) > config.outlier_threshold;
            signal(outlier_mask) = mean(signal(~outlier_mask), 1);
        end
        
        % 3. Filter signal
        if ~strcmp(config.filter_type, 'none')
            signal = apply_filter(signal, config);
        end
        
        % 4. Ensure consistent length (pad or truncate)
        if size(signal, 1) < config.signal_length
            % Pad with zeros
            pad_length = config.signal_length - size(signal, 1);
            signal = [signal; zeros(pad_length, size(signal, 2))];
        else
            % Truncate
            signal = signal(1:config.signal_length, :);
        end
        
        processed_signals{i} = single(signal);
        
        if mod(i, 50) == 0
            fprintf('  Preprocessed %d/%d signals\n', i, length(signals));
        end
    end
    
    info.normalize_method = config.normalize_method;
    info.filter_type = config.filter_type;
    info.total_processed = length(signals);
end

%% Filter Function
function filtered_signal = apply_filter(signal, config)
    %APPLY_FILTER Apply bandpass filter to ECG signal
    
    fs = config.sampling_freq;
    
    % Design filter
    switch config.filter_type
        case 'butterworth'
            % Butterworth bandpass filter
            wp = [config.highpass_freq config.lowpass_freq] / (fs/2);
            ws = [config.highpass_freq/2 config.lowpass_freq*1.5] / (fs/2);
            
            % Constrain frequencies to valid range
            wp = max(0.001, min(0.999, wp));
            ws = max(0.001, min(0.999, ws));
            
            [n, Wn] = buttord(wp, ws, 1, 40);
            [b, a] = butter(n, Wn, 'bandpass');
            
            % Apply filter
            filtered_signal = filtfilt(b, a, signal);
        otherwise
            filtered_signal = signal;
    end
end

%% Segment Signals Function
function [X, y] = segment_signals(signals, labels, config)
    %SEGMENT_SIGNALS Create overlapping segments for CNN training
    %
    % Inputs:
    %   signals - Cell array of preprocessed signals
    %   labels - Signal labels
    %   config - Configuration structure
    %
    % Outputs:
    %   X - Segmented data [samples x leads x segments]
    %   y - Corresponding labels
    
    fprintf('Segmenting signals into overlapping windows...\n');
    
    segment_length = config.segment_length;
    overlap_samples = floor(segment_length * config.segment_overlap);
    stride = segment_length - overlap_samples;
    
    X_segments = [];
    y_segments = [];
    
    for i = 1:length(signals)
        signal = signals{i};
        label = labels(i);
        
        % Create overlapping segments
        num_segments = floor((size(signal, 1) - segment_length) / stride) + 1;
        
        for j = 1:num_segments
            start_idx = (j-1) * stride + 1;
            end_idx = start_idx + segment_length - 1;
            
            if end_idx <= size(signal, 1)
                segment = signal(start_idx:end_idx, :);
                X_segments = cat(3, X_segments, segment);
                y_segments = [y_segments; label];
            end
        end
        
        if mod(i, 50) == 0
            fprintf('  Segmented %d/%d signals\n', i, length(signals));
        end
    end
    
    % Permute to [leads x samples x segments] for CNN
    X = permute(X_segments, [2, 1, 3]);
    y = categorical(y_segments);
    
    fprintf('Total segments created: %d\n', size(X, 3));
    fprintf('Segment shape: [leads=%d, samples=%d, segments=%d]\n', ...
        size(X, 1), size(X, 2), size(X, 3));
end
