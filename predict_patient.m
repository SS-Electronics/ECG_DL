%% ECG INFERENCE - Predict Disease from New Patient ECG
% Load the trained CNN model and classify a new patient's ECG recording
%
% Usage:
%   1. Run train_with_deep_learning.m first (saves ecg_cnn_trained.mat)
%   2. Run this script to predict on any PTB patient
%   3. Or load your own 12-lead ECG data
%
% Requirements:
%   - ecg_cnn_trained.mat (from training)
%   - Deep Learning Toolbox
%   - PTB Database (for loading patient records)

clear; clc; close all;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  ECG CNN Inference - Predict Cardiac Condition                ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% STEP 1: Load Trained Model
fprintf('STEP 1: LOADING TRAINED MODEL\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

model_path = fullfile(pwd, 'ecg_cnn_trained.mat');

if ~isfile(model_path)
    error('Trained model not found at: %s\nRun train_with_deep_learning.m first.', model_path);
end

loaded = load(model_path, 'net');
net = loaded.net;

fprintf('✓ Model loaded from: %s\n\n', model_path);

%% STEP 2: Define Class Labels
class_names = {'Normal', 'MI', 'LBBB', 'RBBB', 'SB', 'AF'};
class_descriptions = {
    'Normal sinus rhythm - Healthy heart'
    'Myocardial infarction - Heart attack'
    'Left bundle branch block'
    'Right bundle branch block'
    'Sinus bradycardia - Slow heart rate'
    'Atrial fibrillation - Irregular rhythm'
};

%% STEP 3: Initialize Data Loader
fprintf('STEP 2: INITIALIZING DATA LOADER\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

ProjectConfig.initialize();

ptb_loader = PTBDataLoader(...
    ProjectConfig.PTB_DB_PATH, ...
    ProjectConfig.SAMPLING_RATE, ...
    ProjectConfig.SIGNAL_LENGTH, ...
    ProjectConfig.NUM_LEADS);

fprintf('✓ PTB Data Loader initialized\n\n');

%% STEP 4: Load a New Patient
fprintf('STEP 3: LOADING PATIENT ECG\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

% ════════════════════════════════════════════════════════════════
%  OPTION A: Load from PTB Database by patient number
% ════════════════════════════════════════════════════════════════

patient_num = 200;  % ← CHANGE THIS to any patient number (1-290)

fprintf('Loading patient %03d from PTB Database...\n', patient_num);

patient_dir = fullfile(ProjectConfig.PTB_DB_PATH, sprintf('patient%03d', patient_num));

if ~isfolder(patient_dir)
    error('Patient directory not found: %s', patient_dir);
end

% Auto-discover record
hea_files = dir(fullfile(patient_dir, '*.hea'));
if isempty(hea_files)
    error('No .hea files found for patient %03d', patient_num);
end

record_name = hea_files(1).name(1:end-4);
fprintf('  Record: %s\n', record_name);

% Load raw signal
try
    dataset = ptb_loader.loadPatientDataset(patient_num, {record_name}, 1);
    raw_signal = mean(dataset.signals, 3);  % Average if multiple records
    fprintf('  Raw shape: (%d, %d)\n', size(raw_signal, 1), size(raw_signal, 2));
catch ME
    error('Failed to load patient: %s', ME.message);
end

% ════════════════════════════════════════════════════════════════
%  OPTION B: Load your own ECG data (uncomment to use)
% ════════════════════════════════════════════════════════════════
%{
% Your data must be: (samples x leads) matrix
% Sampling rate should be 1000 Hz, 12 leads
% If different sampling rate, resample first

raw_signal = load('your_ecg_file.mat');  % Load your data
% Or: raw_signal = csvread('your_ecg.csv');
% Make sure it's (N_samples x 12) format
%}

fprintf('✓ Patient ECG loaded\n\n');

%% STEP 5: Preprocess the Signal
fprintf('STEP 4: PREPROCESSING\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

signal = raw_signal;

% Normalize (z-score per lead)
signal = ptb_loader.normalizeSignal(signal);
fprintf('  ✓ Normalized (z-score per lead)\n');

% Filter (if Signal Processing Toolbox available)
try
    signal = ptb_loader.filterSignal(signal);
    fprintf('  ✓ Bandpass filtered (0.5-40 Hz)\n');
catch
    fprintf('  ⚠ Filtering skipped (Signal Processing Toolbox needed)\n');
end

% Reshape to standard length (10000 x 12)
if size(signal, 1) > ProjectConfig.SIGNAL_LENGTH
    signal = ptb_loader.reshapeSignal(signal, ProjectConfig.SIGNAL_LENGTH);
    fprintf('  ✓ Truncated to %d samples\n', ProjectConfig.SIGNAL_LENGTH);
elseif size(signal, 1) < ProjectConfig.SIGNAL_LENGTH
    signal = ptb_loader.reshapeSignal(signal, ProjectConfig.SIGNAL_LENGTH);
    fprintf('  ✓ Padded to %d samples\n', ProjectConfig.SIGNAL_LENGTH);
else
    fprintf('  ✓ Length already %d samples\n', ProjectConfig.SIGNAL_LENGTH);
end

% Select 12 leads
if size(signal, 2) > 12
    signal = signal(:, 1:12);
    fprintf('  ✓ Selected first 12 leads\n');
end

fprintf('\n  Final shape: (%d, %d)\n\n', size(signal, 1), size(signal, 2));

%% STEP 6: Run Prediction
fprintf('STEP 5: PREDICTION\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

% Wrap in cell + datastore (same format as training)
input_cell = {signal};  % (10000 x 12)
inputDS = arrayDatastore(input_cell, 'OutputType', 'same');

% Get prediction scores
scores = minibatchpredict(net, inputDS);
% probabilities = extractdata(scores);
probabilities = double(scores);

% Convert to class label
[confidence, predicted_idx] = max(probabilities);

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║  DIAGNOSIS RESULT                                         ║\n');
fprintf('╠════════════════════════════════════════════════════════════╣\n');
fprintf('║                                                            ║\n');
fprintf('║  Patient: %03d                                             ║\n', patient_num);
fprintf('║  Predicted: %-45s║\n', class_names{predicted_idx});
fprintf('║  Confidence: %.1f%%                                        ║\n', confidence * 100);
fprintf('║  %s  ║\n', class_descriptions{predicted_idx});
fprintf('║                                                            ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% Print all class probabilities
fprintf('Class Probabilities:\n');
fprintf('────────────────────────────────────────────────────────────────\n');
[sorted_probs, sort_idx] = sort(probabilities, 'descend');
for i = 1:length(class_names)
    idx = sort_idx(i);
    bar_len = round(sorted_probs(i) * 40);
    bar_str = [repmat('#', 1, bar_len), repmat('.', 1, 40 - bar_len)];
    marker = '';
    if idx == predicted_idx
        marker = ' ← PREDICTED';
    end
    fprintf('  %-8s %5.1f%%  |%s|%s\n', ...
        class_names{idx}, sorted_probs(i) * 100, bar_str, marker);
end
fprintf('\n');

%% STEP 7: Parse Actual Diagnosis (for comparison)
fprintf('STEP 6: ACTUAL DIAGNOSIS (from .hea header)\n');
fprintf('────────────────────────────────────────────────────────────────\n\n');

hea_path = fullfile(patient_dir, hea_files(1).name);
fid = fopen(hea_path, 'r');
actual_diagnosis = 'Unknown';
actual_age = 'Unknown';
actual_sex = 'Unknown';

if fid ~= -1
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            if contains(line, 'Reason for admission', 'IgnoreCase', true)
                parts = strsplit(line, ':');
                if length(parts) >= 2
                    actual_diagnosis = strtrim(strjoin(parts(2:end), ':'));
                end
            end
            if contains(line, '<age>', 'IgnoreCase', true)
                parts = strsplit(line, ':');
                if length(parts) >= 2
                    actual_age = strtrim(parts{2});
                end
            end
            if contains(line, '<sex>', 'IgnoreCase', true)
                parts = strsplit(line, ':');
                if length(parts) >= 2
                    actual_sex = strtrim(parts{2});
                end
            end
        end
    end
    fclose(fid);
end

fprintf('  Patient: %03d\n', patient_num);
fprintf('  Age: %s\n', actual_age);
fprintf('  Sex: %s\n', actual_sex);
fprintf('  Actual diagnosis: %s\n', actual_diagnosis);
fprintf('  Model prediction: %s (%.1f%%)\n\n', class_names{predicted_idx}, confidence * 100);

%% STEP 8: Batch Prediction on Multiple Patients
fprintf('STEP 7: BATCH PREDICTION (optional)\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');

fprintf('To predict on multiple patients, use this pattern:\n\n');
fprintf('  patient_list = [10, 42, 100, 150, 200];\n');
fprintf('  for p = patient_list\n');
fprintf('      %% Load, preprocess, predict (same as above)\n');
fprintf('  end\n\n');

% Uncomment below to run batch prediction:
%{
patient_list = [10, 42, 100, 150, 200, 250];

fprintf('Batch predicting %d patients...\n\n', length(patient_list));
fprintf('%-10s %-15s %-8s %s\n', 'Patient', 'Prediction', 'Conf.', 'Actual Diagnosis');
fprintf('%s\n', repmat('-', 1, 65));

for p_idx = 1:length(patient_list)
    pnum = patient_list(p_idx);
    pdir = fullfile(ProjectConfig.PTB_DB_PATH, sprintf('patient%03d', pnum));
    
    if ~isfolder(pdir), continue; end
    
    hfiles = dir(fullfile(pdir, '*.hea'));
    if isempty(hfiles), continue; end
    
    rname = hfiles(1).name(1:end-4);
    
    try
        ds = ptb_loader.loadPatientDataset(pnum, {rname}, 1);
        sig = mean(ds.signals, 3);
        sig = ptb_loader.normalizeSignal(sig);
        
        if size(sig, 1) ~= ProjectConfig.SIGNAL_LENGTH
            sig = ptb_loader.reshapeSignal(sig, ProjectConfig.SIGNAL_LENGTH);
        end
        if size(sig, 2) > 12
            sig = sig(:, 1:12);
        end
        
        inp = {sig};
        inDS = arrayDatastore(inp, 'OutputType', 'same');
        sc = minibatchpredict(net, inDS);
        pr = extractdata(sc);
        [conf, pidx] = max(pr);
        
        % Get actual diagnosis
        fid2 = fopen(fullfile(pdir, hfiles(1).name), 'r');
        adiag = 'Unknown';
        while ~feof(fid2)
            ln = fgetl(fid2);
            if ischar(ln) && contains(ln, 'Reason for admission', 'IgnoreCase', true)
                pts = strsplit(ln, ':');
                if length(pts) >= 2
                    adiag = strtrim(strjoin(pts(2:end), ':'));
                end
            end
        end
        fclose(fid2);
        
        fprintf('%-10s %-15s %5.1f%%   %s\n', ...
            sprintf('P%03d', pnum), class_names{pidx}, conf*100, adiag);
    catch
        fprintf('%-10s %-15s %8s %s\n', sprintf('P%03d', pnum), 'FAILED', '', '');
    end
end
fprintf('\n');
%}

fprintf('════════════════════════════════════════════════════════════════\n');
fprintf(' INFERENCE COMPLETE\n');
fprintf('════════════════════════════════════════════════════════════════\n\n');