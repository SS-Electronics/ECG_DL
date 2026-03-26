%% Load PTB Database - Modular Data Loading Function
% Consolidated loading function with:
%   - Auto-discovery of .hea/.dat record files per patient
%   - Real diagnosis parsing from .hea headers
%   - Preprocessing (normalize, filter, reshape)
%   - Class distribution reporting
%
% Usage:
%   [data, labels, info] = load_ptb_database(ptb_path, ptb_loader)
%   [data, labels, info] = load_ptb_database(ptb_path, ptb_loader, 'MaxPatients', 100)
%   [data, labels, info] = load_ptb_database(ptb_path, ptb_loader, 'Filter', true)
%
% Inputs:
%   ptb_path   - Path to PTB Database root (contains patient001/, patient002/, ...)
%   ptb_loader - Initialized PTBDataLoader object
%
% Optional Name-Value Pairs:
%   'MaxPatients'  - Maximum patients to load (default: all)
%   'Filter'       - Apply bandpass filter (default: true)
%   'Verbose'      - Print progress (default: true)
%
% Outputs:
%   data   - (10000 x 12 x N) preprocessed ECG tensor
%   labels - (N x 6) one-hot encoded labels
%   info   - struct with loading statistics and class distribution

function [data, labels, info] = load_ptb_database(ptb_path, ptb_loader, varargin)

    %% Parse optional arguments
    p = inputParser;
    addRequired(p, 'ptb_path', @ischar);
    addRequired(p, 'ptb_loader');
    addParameter(p, 'MaxPatients', Inf, @isnumeric);
    addParameter(p, 'Filter', true, @islogical);
    addParameter(p, 'Verbose', true, @islogical);
    parse(p, ptb_path, ptb_loader, varargin{:});
    
    opts = p.Results;
    verbose = opts.Verbose;

    %% Discover patient directories
    patient_dirs = dir(fullfile(ptb_path, 'patient*'));
    
    if isempty(patient_dirs)
        error('load_ptb_database:noPatients', ...
            'No patient folders found in: %s', ptb_path);
    end
    
    max_patients = min(opts.MaxPatients, length(patient_dirs));
    
    if verbose
        fprintf('[load_ptb_database] Found %d patient folders, loading up to %d\n\n', ...
            length(patient_dirs), max_patients);
    end

    %% Initialize outputs
    data = [];
    labels = [];
    
    info = struct();
    info.total_dirs = length(patient_dirs);
    info.attempted = 0;
    info.loaded = 0;
    info.skipped_no_hea = 0;
    info.skipped_unknown_dx = 0;
    info.skipped_load_error = 0;
    info.class_names = {'Normal', 'MI', 'LBBB', 'RBBB', 'SB', 'AF'};
    info.class_counts = zeros(1, 6);
    info.patient_list = [];
    info.diagnosis_strings = {};

    %% Load each patient
    for p = 1:max_patients
        info.attempted = info.attempted + 1;
        patient_folder = fullfile(ptb_path, patient_dirs(p).name);
        
        % --- Auto-discover .hea files ---
        hea_files = dir(fullfile(patient_folder, '*.hea'));
        
        if isempty(hea_files)
            info.skipped_no_hea = info.skipped_no_hea + 1;
            if verbose
                fprintf('  [%d/%d] %s: No .hea files, skipping\n', ...
                    p, max_patients, patient_dirs(p).name);
            end
            continue;
        end
        
        % Use first record
        record_name = hea_files(1).name(1:end-4);
        hea_filepath = fullfile(patient_folder, hea_files(1).name);
        
        % Extract patient number from folder name
        patient_num = str2double(regexp(patient_dirs(p).name, '\d+', 'match', 'once'));
        
        % --- Parse real diagnosis from .hea header ---
        [disease_class, dx_string] = parse_diagnosis(hea_filepath);
        
        if disease_class == 0
            info.skipped_unknown_dx = info.skipped_unknown_dx + 1;
            if verbose
                fprintf('  [%d/%d] %s: Unknown diagnosis (%s), skipping\n', ...
                    p, max_patients, patient_dirs(p).name, dx_string);
            end
            continue;
        end
        
        % --- Load and preprocess ---
        try
            dataset = ptb_loader.loadPatientDataset(patient_num, {record_name}, disease_class);
            
            if dataset.num_records == 0
                info.skipped_load_error = info.skipped_load_error + 1;
                continue;
            end
            
            % Average across records if multiple exist
            patient_signal = mean(dataset.signals, 3);
            
            % Optional: apply bandpass filter
            if opts.Filter
                try
                    [b, a] = butter(4, [0.5 40] / (ptb_loader.sampling_rate / 2), 'bandpass');
                    for lead = 1:size(patient_signal, 2)
                        patient_signal(:, lead) = filtfilt(b, a, double(patient_signal(:, lead)));
                    end
                catch
                    % Filter failed silently — use unfiltered signal
                end
            end
            
            % Stack into output arrays
            data = cat(3, data, patient_signal);
            
            one_hot = zeros(1, 6);
            one_hot(disease_class) = 1;
            labels = [labels; one_hot];
            
            % Update stats
            info.loaded = info.loaded + 1;
            info.class_counts(disease_class) = info.class_counts(disease_class) + 1;
            info.patient_list(end+1) = patient_num;
            info.diagnosis_strings{end+1} = dx_string;
            
            if verbose
                fprintf('  [%d/%d] %s -> %s  class %d (%s)\n', ...
                    p, max_patients, patient_dirs(p).name, record_name, ...
                    disease_class, info.class_names{disease_class});
            end
            
        catch ME
            info.skipped_load_error = info.skipped_load_error + 1;
            if verbose
                fprintf('  [%d/%d] %s -> FAILED (%s)\n', ...
                    p, max_patients, patient_dirs(p).name, ME.message);
            end
        end
    end

    %% Print summary
    if verbose
        fprintf('\n');
        fprintf('╔════════════════════════════════════════════════════════════╗\n');
        fprintf('║  PTB Database Loading Summary                             ║\n');
        fprintf('╠════════════════════════════════════════════════════════════╣\n');
        fprintf('║  Attempted: %-4d patients                                 ║\n', info.attempted);
        fprintf('║  Loaded:    %-4d patients                                 ║\n', info.loaded);
        fprintf('║  Skipped:   %-4d (no .hea: %d, unknown dx: %d, error: %d) ║\n', ...
            info.skipped_no_hea + info.skipped_unknown_dx + info.skipped_load_error, ...
            info.skipped_no_hea, info.skipped_unknown_dx, info.skipped_load_error);
        fprintf('╠════════════════════════════════════════════════════════════╣\n');
        fprintf('║  Class distribution:                                      ║\n');
        for c = 1:6
            fprintf('║    %-8s: %-4d samples                                  ║\n', ...
                info.class_names{c}, info.class_counts(c));
        end
        fprintf('║                                                            ║\n');
        if info.loaded > 0
            fprintf('║  Data shape: (%d x %d x %d)                             ║\n', ...
                size(data, 1), size(data, 2), size(data, 3));
        end
        fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    end
    
    % Sanity check
    if info.loaded < 2
        warning('load_ptb_database:tooFew', ...
            'Only %d patients loaded. Check PTB_DB_PATH and .hea files.', info.loaded);
    end
end


%% ═══════════════════════════════════════════════════════════════════════
%  DIAGNOSIS PARSER - Reads "Reason for admission" from .hea header
%  Returns: disease_class (1-6, or 0 if unknown), raw diagnosis string
%% ═══════════════════════════════════════════════════════════════════════

function [disease_class, dx_string] = parse_diagnosis(hea_filepath)
    disease_class = 0;
    dx_string = 'Not found';
    
    fid = fopen(hea_filepath, 'r');
    if fid == -1, return; end
    
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line), continue; end
        
        if contains(line, 'Reason for admission', 'IgnoreCase', true)
            % Extract the diagnosis text after the colon
            parts = strsplit(line, ':');
            if length(parts) >= 2
                dx_string = strtrim(strjoin(parts(2:end), ':'));
            else
                dx_string = strtrim(line);
            end
            
            % Map diagnosis string to class ID
            disease_class = map_diagnosis_to_class(dx_string);
            break;
        end
    end
    
    fclose(fid);
end


%% ═══════════════════════════════════════════════════════════════════════
%  DIAGNOSIS MAPPER - Maps diagnosis string to class 1-6
%  Covers known PTB diagnosis variants
%% ═══════════════════════════════════════════════════════════════════════

function class_id = map_diagnosis_to_class(dx_string)
    dx = lower(strtrim(dx_string));
    class_id = 0;  % Unknown by default
    
    % ── Class 1: Normal / Healthy ──
    if contains(dx, 'healthy') || ...
       contains(dx, 'normal') || ...
       contains(dx, 'n/a') || ...
       strcmp(dx, 'heart failure (nyha 1)')  % NYHA 1 is functionally normal
        class_id = 1;
        return;
    end
    
    % ── Class 2: Myocardial Infarction ──
    if contains(dx, 'myocardial infarction') || ...
       contains(dx, 'infarction') || ...
       contains(dx, 'mi')
        class_id = 2;
        return;
    end
    
    % ── Class 3: Left Bundle Branch Block ──
    if (contains(dx, 'bundle branch block') && contains(dx, 'left')) || ...
       contains(dx, 'lbbb')
        class_id = 3;
        return;
    end
    
    % ── Class 4: Right Bundle Branch Block ──
    if (contains(dx, 'bundle branch block') && contains(dx, 'right')) || ...
       contains(dx, 'rbbb')
        class_id = 4;
        return;
    end
    
    % ── Class 5: Sinus Bradycardia / slow rate ──
    if contains(dx, 'bradycardia') || ...
       contains(dx, 'sinus node') || ...
       contains(dx, 'sick sinus')
        class_id = 5;
        return;
    end
    
    % ── Class 6: Atrial Fibrillation / Dysrhythmia ──
    if contains(dx, 'atrial fibrillation') || ...
       contains(dx, 'atrial flutter') || ...
       contains(dx, 'dysrhythmia') || ...
       contains(dx, 'tachycardia') || ...
       contains(dx, 'palpitation')
        class_id = 6;
        return;
    end
    
    % ── Broader catch-alls (lower priority) ──
    
    % Cardiomyopathy / heart failure → group with MI (structural heart disease)
    if contains(dx, 'cardiomyopathy') || ...
       contains(dx, 'heart failure') || ...
       contains(dx, 'hypertrop') || ...
       contains(dx, 'myocarditis') || ...
       contains(dx, 'unstable angina') || ...
       contains(dx, 'coronary')
        class_id = 2;
        return;
    end
    
    % Valvular disease → group with Normal (often structurally normal ECG)
    if contains(dx, 'valvular') || ...
       contains(dx, 'stable angina')
        class_id = 1;
        return;
    end
    
    % If nothing matched, class_id stays 0 (will be skipped)
end