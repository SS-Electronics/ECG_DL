%% ECG_CNN_Classification - Setup Script
% Run this FIRST to add all project paths to MATLAB
% Then you can run main.m or examples.m

clear; clc;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║   ECG CNN Classification Framework - Setup                 ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% Get the directory where this script is located
project_root = fileparts(mfilename('fullpath'));

fprintf('[Setup] Project root: %s\n', project_root);

% Define all subdirectories to add to path
subdirs = {
    'config'
    'data'
    'models'
    'core'
    'training'
    'utils'
    'visualization'
};

% Add each subdirectory to MATLAB path
fprintf('[Setup] Adding directories to MATLAB path:\n');
for i = 1:length(subdirs)
    dir_path = fullfile(project_root, subdirs{i});
    if isfolder(dir_path)
        addpath(dir_path);
        fprintf('  ✓ Added: %s\n', subdirs{i});
    else
        fprintf('  ✗ Not found: %s\n', subdirs{i});
    end
end

% Also add project root
addpath(project_root);
fprintf('  ✓ Added: Project root\n');

fprintf('\n[Setup] Checking for required files...\n');

% Check for key files
key_files = {
    'config/ProjectConfig.m'
    'data/DataLoader.m'
    'models/BaseModel.m'
    'models/CNN1D.m'
    'models/ResNetECG.m'
    'core/ModelManager.m'
    'training/TrainingOrchestrator.m'
};

all_found = true;
for i = 1:length(key_files)
    file_path = fullfile(project_root, key_files{i});
    if isfile(file_path)
        fprintf('  ✓ Found: %s\n', key_files{i});
    else
        fprintf('  ✗ Missing: %s\n', key_files{i});
        all_found = false;
    end
end

fprintf('\n');

if all_found
    fprintf('[Setup] ✅ All files found!\n');
    fprintf('[Setup] ✅ All paths added!\n\n');
    
    fprintf('════════════════════════════════════════════════════════════\n');
    fprintf('Setup completed successfully!\n');
    fprintf('════════════════════════════════════════════════════════════\n\n');
    
    fprintf('You can now run:\n');
    fprintf('  >> main.m          (Run complete automated training)\n');
    fprintf('  >> examples.m      (Learn by example)\n');
    fprintf('  >> ProjectConfig.initialize()  (Initialize project)\n\n');
    
else
    fprintf('[Setup] ❌ Some files are missing!\n');
    fprintf('[Setup] Make sure you extracted all files from the ZIP\n\n');
end

% Save the path permanently
fprintf('Saving MATLAB path...\n');
try
    savepath;
    fprintf('✓ MATLAB path saved permanently\n\n');
catch
    fprintf('⚠ Could not save path permanently (but it''s added for this session)\n');
    fprintf('  To make it permanent, go to: Home > Set Path > Save\n\n');
end

fprintf('════════════════════════════════════════════════════════════\n');
fprintf('Setup Complete! Ready to start training ECG CNN models\n');
fprintf('════════════════════════════════════════════════════════════\n\n');
