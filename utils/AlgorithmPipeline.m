classdef AlgorithmPipeline < handle
    % AlgorithmPipeline - Base class for composable signal processing algorithms
    % Allows chaining different algorithms in a pipeline
    
    properties
        name                % Pipeline name
        algorithms          % Cell array of algorithm handles
        config              % Configuration struct
        execution_log       % Log of executed algorithms
    end
    
    methods
        function obj = AlgorithmPipeline(pipeline_name)
            % Initialize empty pipeline
            obj.name = pipeline_name;
            obj.algorithms = {};
            obj.config = struct();
            obj.execution_log = struct();
            obj.execution_log.steps = {};
            obj.execution_log.timestamps = [];
        end
        
        function addAlgorithm(obj, algorithm_name, algorithm_func, params)
            % Add an algorithm to the pipeline
            % algorithm_name: Unique identifier
            % algorithm_func: Function handle
            % params: Parameter struct for the algorithm
            
            idx = length(obj.algorithms) + 1;
            obj.algorithms{idx}.name = algorithm_name;
            obj.algorithms{idx}.func = algorithm_func;
            obj.algorithms{idx}.params = params;
            
            fprintf('[%s] Added algorithm: %s\n', obj.name, algorithm_name);
        end
        
        function output = execute(obj, input_data, varargin)
            % Execute pipeline on input data
            % Optional: specify subset of algorithms
            
            if nargin > 2
                algorithm_indices = varargin{1};
            else
                algorithm_indices = 1:length(obj.algorithms);
            end
            
            output = input_data;
            
            fprintf('[%s] Executing pipeline with %d algorithms...\n', ...
                obj.name, length(algorithm_indices));
            
            for i = algorithm_indices
                algo = obj.algorithms{i};
                
                try
                    tic;
                    output = algo.func(output, algo.params);
                    elapsed = toc;
                    
                    obj.execution_log.steps{end+1} = algo.name;
                    obj.execution_log.timestamps(end+1) = elapsed;
                    
                    fprintf('[%s] ✓ %s (%.3fs)\n', obj.name, algo.name, elapsed);
                    
                catch ME
                    fprintf('[%s] ✗ %s failed: %s\n', obj.name, algo.name, ME.message);
                    error('Pipeline execution failed at %s', algo.name);
                end
            end
        end
        
        function printPipeline(obj)
            % Print pipeline structure
            fprintf('\n===== Pipeline: %s =====\n', obj.name);
            for i = 1:length(obj.algorithms)
                algo = obj.algorithms{i};
                fprintf('%d. %s\n', i, algo.name);
                fprintf('   Params: %d fields\n', length(fieldnames(algo.params)));
            end
            fprintf('========================\n\n');
        end
    end
end

%% Built-in Algorithm Functions

function output = normalize_signal(input_data, params)
    % Normalize signal using z-score
    mean_val = mean(input_data(:));
    std_val = std(input_data(:));
    output = (input_data - mean_val) / std_val;
end

function output = filter_signal(input_data, params)
    % Bandpass filter
    Fs = params.fs;
    Fpass = params.fpass;
    
    d = designfilt('bandpass', ...
        'PassbandFrequency1', Fpass(1), ...
        'PassbandFrequency2', Fpass(2), ...
        'SampleRate', Fs);
    
    if size(input_data, 2) > 1
        output = zeros(size(input_data));
        for lead = 1:size(input_data, 2)
            output(:, lead) = filtfilt(d, input_data(:, lead));
        end
    else
        output = filtfilt(d, input_data);
    end
end

function output = denoise_signal(input_data, params)
    % Denoise using wavelet transform
    [C, L] = wavedec(input_data, params.level, params.wavelet);
    sigma = median(abs(C(L(1)+1:L(1)+L(2))))/0.6745;
    thr = sigma * sqrt(2 * log(length(input_data)));
    C_thresh = wthresh(C, 's', thr);
    output = waverec(C_thresh, L, params.wavelet);
    output = output(1:size(input_data, 1), :);
end

function output = augment_signal(input_data, params)
    % Data augmentation
    output = input_data;
    
    % Scaling
    if isfield(params, 'scaling_factor')
        output = output * params.scaling_factor;
    end
    
    % Noise addition
    if isfield(params, 'noise_level')
        noise = randn(size(output)) * params.noise_level;
        output = output + noise;
    end
    
    % Time stretching
    if isfield(params, 'stretch_factor')
        output = interp1(1:size(output,1), output, ...
            linspace(1, size(output,1), floor(size(output,1) * params.stretch_factor)));
    end
end

function output = extract_features(input_data, params)
    % Extract statistical features
    if isa(input_data, 'double')
        output = input_data;  % Pass through if already features
    else
        output = input_data;
    end
end
