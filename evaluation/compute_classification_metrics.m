%% Compute Classification Metrics Function
function metrics = compute_classification_metrics(y_true, y_pred)
    %COMPUTE_CLASSIFICATION_METRICS Calculate comprehensive classification metrics
    %
    % Inputs:
    %   y_true - True labels (categorical)
    %   y_pred - Predicted labels (categorical)
    %
    % Outputs:
    %   metrics - Structure containing various metrics
    
    % Convert to categorical if needed
    if ~iscategorical(y_true)
        y_true = categorical(y_true);
    end
    if ~iscategorical(y_pred)
        y_pred = categorical(y_pred);
    end
    
    % Accuracy
    correct = sum(y_true == y_pred);
    total = length(y_true);
    metrics.accuracy = correct / total;
    
    % Convert to numeric for confusion matrix
    y_true_num = double(y_true);
    y_pred_num = double(y_pred);
    
    % Confusion Matrix
    cm = confusionmat(y_true_num, y_pred_num);
    metrics.confusion_matrix = cm;
    
    % Per-class metrics
    n_classes = size(cm, 1);
    precision_per_class = zeros(n_classes, 1);
    recall_per_class = zeros(n_classes, 1);
    f1_per_class = zeros(n_classes, 1);
    
    for i = 1:n_classes
        tp = cm(i, i);
        fp = sum(cm(:, i)) - tp;
        fn = sum(cm(i, :)) - tp;
        
        % Precision
        if (tp + fp) > 0
            precision_per_class(i) = tp / (tp + fp);
        else
            precision_per_class(i) = 0;
        end
        
        % Recall
        if (tp + fn) > 0
            recall_per_class(i) = tp / (tp + fn);
        else
            recall_per_class(i) = 0;
        end
        
        % F1-Score
        if (precision_per_class(i) + recall_per_class(i)) > 0
            f1_per_class(i) = 2 * (precision_per_class(i) * recall_per_class(i)) / ...
                             (precision_per_class(i) + recall_per_class(i));
        else
            f1_per_class(i) = 0;
        end
    end
    
    % Macro averages
    metrics.precision = mean(precision_per_class);
    metrics.recall = mean(recall_per_class);
    metrics.f1_score = mean(f1_per_class);
    
    % Store per-class metrics
    metrics.precision_per_class = precision_per_class;
    metrics.recall_per_class = recall_per_class;
    metrics.f1_per_class = f1_per_class;
    
    % Specificity and sensitivity
    metrics.sensitivity = metrics.recall;  % Same as recall
    metrics.specificity = zeros(n_classes, 1);
    for i = 1:n_classes
        tn = sum(sum(cm)) - sum(cm(i, :)) - sum(cm(:, i)) + cm(i, i);
        fp = sum(cm(:, i)) - cm(i, i);
        if (tn + fp) > 0
            metrics.specificity(i) = tn / (tn + fp);
        else
            metrics.specificity(i) = 0;
        end
    end
    
    % Weighted metrics
    class_counts = sum(cm, 2);
    total_samples = sum(class_counts);
    weights = class_counts / total_samples;
    
    metrics.precision_weighted = sum(precision_per_class .* weights);
    metrics.recall_weighted = sum(recall_per_class .* weights);
    metrics.f1_weighted = sum(f1_per_class .* weights);
    
end

%% ROC Analysis Function
function [fpr, tpr, thresholds, auc_value] = roc_analysis(y_true, y_pred)
    %ROC_ANALYSIS Compute ROC curve and AUC for binary classification
    %
    % Inputs:
    %   y_true - True binary labels
    %   y_pred - Predicted probabilities or labels
    %
    % Outputs:
    %   fpr - False Positive Rate
    %   tpr - True Positive Rate
    %   thresholds - Threshold values
    %   auc_value - Area Under the Curve
    
    % Handle categorical
    if iscategorical(y_true)
        y_true = double(y_true);
    end
    if iscategorical(y_pred)
        y_pred = double(y_pred);
    end
    
    % Get unique values
    unique_vals = unique(y_true);
    
    % Make binary
    pos_class = unique_vals(end);
    y_true_binary = (y_true == pos_class);
    y_pred_binary = (y_pred == pos_class);
    
    % Calculate metrics at different thresholds
    n_thresholds = 100;
    thresholds = linspace(0, 1, n_thresholds);
    
    fpr = zeros(n_thresholds, 1);
    tpr = zeros(n_thresholds, 1);
    
    for i = 1:n_thresholds
        threshold = thresholds(i);
        
        % Predictions at this threshold
        y_pred_thresh = (y_pred_binary >= threshold);
        
        % Calculate FPR and TPR
        tp = sum(y_pred_thresh & y_true_binary);
        fp = sum(y_pred_thresh & ~y_true_binary);
        tn = sum(~y_pred_thresh & ~y_true_binary);
        fn = sum(~y_pred_thresh & y_true_binary);
        
        if (tp + fn) > 0
            tpr(i) = tp / (tp + fn);
        else
            tpr(i) = 0;
        end
        
        if (fp + tn) > 0
            fpr(i) = fp / (fp + tn);
        else
            fpr(i) = 0;
        end
    end
    
    % Calculate AUC using trapezoid rule
    auc_value = trapz(fpr, tpr);
end

%% Precision-Recall Curve
function [precision, recall, auc_pr] = precision_recall_curve(y_true, y_pred_proba)
    %PRECISION_RECALL_CURVE Compute precision-recall curve
    
    if iscategorical(y_true)
        y_true = double(y_true);
    end
    
    unique_vals = unique(y_true);
    pos_class = unique_vals(end);
    
    % Sort by probability
    [sorted_proba, idx] = sort(y_pred_proba, 'descend');
    sorted_true = y_true(idx) == pos_class;
    
    % Calculate precision and recall
    n_samples = length(y_true);
    n_positives = sum(y_true == pos_class);
    
    tp = cumsum(sorted_true);
    fp = cumsum(~sorted_true);
    
    precision = tp ./ (tp + fp);
    recall = tp / n_positives;
    
    % AUC for precision-recall
    auc_pr = trapz(recall, precision);
end

%% Kappa Statistics (Inter-rater agreement)
function kappa = cohen_kappa(y_true, y_pred)
    %COHEN_KAPPA Compute Cohen's Kappa statistic
    %
    % Measures agreement accounting for chance
    
    if iscategorical(y_true)
        y_true = double(y_true);
    end
    if iscategorical(y_pred)
        y_pred = double(y_pred);
    end
    
    % Observed agreement
    n = length(y_true);
    po = sum(y_true == y_pred) / n;
    
    % Expected agreement
    unique_classes = unique([y_true; y_pred]);
    pe = 0;
    for c = unique_classes'
        p_true = sum(y_true == c) / n;
        p_pred = sum(y_pred == c) / n;
        pe = pe + p_true * p_pred;
    end
    
    % Cohen's Kappa
    if pe == 1
        kappa = 1;
    else
        kappa = (po - pe) / (1 - pe);
    end
end

%% Sensitivity and Specificity Analysis
function analysis = sensitivity_specificity_analysis(y_true, y_pred)
    %SENSITIVITY_SPECIFICITY_ANALYSIS Detailed analysis per class
    
    if iscategorical(y_true)
        y_true = double(y_true);
    end
    if iscategorical(y_pred)
        y_pred = double(y_pred);
    end
    
    unique_classes = unique(y_true);
    n_classes = length(unique_classes);
    
    analysis.sensitivity = zeros(n_classes, 1);
    analysis.specificity = zeros(n_classes, 1);
    analysis.youden_index = zeros(n_classes, 1);
    
    for i = 1:n_classes
        class = unique_classes(i);
        
        % True positive and false negative
        tp = sum((y_true == class) & (y_pred == class));
        fn = sum((y_true == class) & (y_pred ~= class));
        
        % True negative and false positive
        tn = sum((y_true ~= class) & (y_pred ~= class));
        fp = sum((y_true ~= class) & (y_pred == class));
        
        % Sensitivity (Recall)
        if (tp + fn) > 0
            analysis.sensitivity(i) = tp / (tp + fn);
        else
            analysis.sensitivity(i) = 0;
        end
        
        % Specificity
        if (tn + fp) > 0
            analysis.specificity(i) = tn / (tn + fp);
        else
            analysis.specificity(i) = 0;
        end
        
        % Youden Index (measure of diagnostic accuracy)
        analysis.youden_index(i) = analysis.sensitivity(i) + analysis.specificity(i) - 1;
    end
    
    analysis.classes = unique_classes;
end

%% Cross-Validation Metrics
function cv_metrics = cross_validation_analysis(y_true, y_pred, k_folds)
    %CROSS_VALIDATION_ANALYSIS Analyze cross-validation results
    
    n_samples = length(y_true);
    fold_size = floor(n_samples / k_folds);
    
    accuracies = zeros(k_folds, 1);
    f1_scores = zeros(k_folds, 1);
    
    for fold = 1:k_folds
        test_idx = (fold-1)*fold_size+1 : min(fold*fold_size, n_samples);
        
        y_test = y_true(test_idx);
        y_pred_fold = y_pred(test_idx);
        
        metrics = compute_classification_metrics(y_test, y_pred_fold);
        accuracies(fold) = metrics.accuracy;
        f1_scores(fold) = metrics.f1_score;
    end
    
    cv_metrics.accuracies = accuracies;
    cv_metrics.f1_scores = f1_scores;
    cv_metrics.mean_accuracy = mean(accuracies);
    cv_metrics.std_accuracy = std(accuracies);
    cv_metrics.mean_f1 = mean(f1_scores);
    cv_metrics.std_f1 = std(f1_scores);
end

%% Print Metrics Report
function print_metrics_report(metrics)
    %PRINT_METRICS_REPORT Display formatted metrics report
    
    fprintf('\n========================================\n');
    fprintf('CLASSIFICATION METRICS REPORT\n');
    fprintf('========================================\n\n');
    
    fprintf('Overall Metrics:\n');
    fprintf('  Accuracy:  %.4f\n', metrics.accuracy);
    fprintf('  Precision: %.4f\n', metrics.precision);
    fprintf('  Recall:    %.4f\n', metrics.recall);
    fprintf('  F1-Score:  %.4f\n\n', metrics.f1_score);
    
    fprintf('Weighted Metrics:\n');
    fprintf('  Precision: %.4f\n', metrics.precision_weighted);
    fprintf('  Recall:    %.4f\n', metrics.recall_weighted);
    fprintf('  F1-Score:  %.4f\n\n', metrics.f1_weighted);
    
    fprintf('Confusion Matrix:\n');
    disp(metrics.confusion_matrix);
    fprintf('\n');
end
