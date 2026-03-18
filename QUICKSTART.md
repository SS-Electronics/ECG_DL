# ECG CNN Classification Framework - Quick Start Guide

## 🚀 5-Minute Setup

### 1. Initialize Project
```matlab
cd ECG_CNN_Classification/
ProjectConfig.initialize();
```

### 2. Create Orchestrator
```matlab
orchestrator = TrainingOrchestrator(ProjectConfig);
```

### 3. Prepare Data
```matlab
% Load your data (10000 samples, 12 leads, one-hot labels)
orchestrator.data = your_train_data;        % (10000 x 12 x n_samples)
orchestrator.labels = your_train_labels;    % (n_samples x 6)
```

### 4. Train Models
```matlab
models = {'CNN1D', 'ResNetECG'};
orchestrator.trainMultipleModels(models);
```

### 5. View Results
```matlab
orchestrator.plotTrainingResults();
orchestrator.model_manager.compareModels(models);
```

---

## 📊 Key Classes

| Class | Purpose | Location |
|-------|---------|----------|
| `ProjectConfig` | Global configuration | `config/ProjectConfig.m` |
| `DataLoader` | Data loading & preprocessing | `data/DataLoader.m` |
| `BaseModel` | Abstract model class | `models/BaseModel.m` |
| `CNN1D` | 1D CNN implementation | `models/CNN1D.m` |
| `ResNetECG` | ResNet-inspired model | `models/ResNetECG.m` |
| `ModelManager` | Model registration & management | `core/ModelManager.m` |
| `TrainingOrchestrator` | Automated training pipeline | `training/TrainingOrchestrator.m` |
| `AlgorithmPipeline` | Composable algorithms | `utils/AlgorithmPipeline.m` |

---

## ⚙️ Configuration Quick Reference

```matlab
% Global settings
ProjectConfig.SAMPLING_RATE        % 1000 Hz
ProjectConfig.SIGNAL_LENGTH        % 10000 samples
ProjectConfig.NUM_LEADS            % 12 leads
ProjectConfig.BATCH_SIZE           % 32
ProjectConfig.NUM_EPOCHS           % 100
ProjectConfig.LEARNING_RATE        % 0.001

% Per-model customization
custom_params = struct(...
    'dropout_rate', 0.4, ...
    'l2_regularization', 1e-4);
```

---

## 📈 Common Tasks

### Train Single Model
```matlab
orchestrator.trainSingleModel('CNN1D');
```

### Train Multiple Models
```matlab
models = {'CNN1D', 'ResNetECG'};
orchestrator.trainMultipleModels(models);
```

### Make Predictions
```matlab
sample_ecg = randn(10000, 12);
predictions = predict(model, sample_ecg);
[confidence, class_id] = max(predictions);
fprintf('Class: %s\n', ProjectConfig.DISEASE_LABELS{class_id});
```

### Save/Load Models
```matlab
% Save
save('models/trained/my_model.mat', 'model');

% Load
loaded = load('models/trained/my_model.mat');
model = loaded.model;
```

### Compare Models
```matlab
orchestrator.model_manager.compareModels({'CNN1D', 'ResNetECG'});
orchestrator.model_manager.plotComparison({'CNN1D', 'ResNetECG'});
```

---

## 🧬 Disease Classes

```
1: Normal
2: MI (Myocardial Infarction)
3: LBBB (Left Bundle Branch Block)
4: RBBB (Right Bundle Branch Block)
5: SB (Sinus Bradycardia)
6: AF (Atrial Fibrillation)
```

---

## 📁 Output Files

| Location | Contents |
|----------|----------|
| `models/trained/` | Trained model weights |
| `results/training_summary.txt` | Training report |
| `results/plots/` | Visualizations |
| `results/metrics/` | Performance metrics |

---

## 🔧 Adding a New Model

### 1. Create Model Class
```matlab
classdef MyModel < BaseModel
    methods
        function obj = MyModel(params)
            obj = obj@BaseModel('MyModel', params);
            obj.createArchitecture();
        end
        
        function createArchitecture(obj)
            % Define architecture
        end
        
        function output = predict(obj, input_data)
            % Forward pass
        end
    end
end
```

### 2. Register in ModelManager
```matlab
% In ModelManager.registerDefaultModels()
obj.registerModel('MyModel', 'Description', 'MyModel', params);
```

### 3. Train
```matlab
orchestrator.trainMultipleModels({'MyModel'});
```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "rdsamp not found" | `ProjectConfig.initialize();` |
| Out of memory | Reduce `BATCH_SIZE` or sample count |
| Poor accuracy | Increase epochs, adjust learning rate |
| Training slow | Increase batch size, reduce samples |

---

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **Examples**: Run `examples.m`
- **Main Script**: See `main.m`
- **Class Documentation**: Check docstrings in each .m file

---

## 🎯 Workflow Summary

```
1. ProjectConfig.initialize()
   ↓
2. Create TrainingOrchestrator(ProjectConfig)
   ↓
3. Load data into orchestrator
   ↓
4. trainMultipleModels({'CNN1D', 'ResNetECG'})
   ↓
5. plotTrainingResults()
   ↓
6. compareModels()
   ↓
7. Use predict() for inference
```

---

**Last Updated**: March 2024  
**Framework Version**: 1.0.0
