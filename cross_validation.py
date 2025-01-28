# models/cross_validation.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import wandb
from typing import Dict, Any, Optional
import time
from torch.utils.data import DataLoader, TensorDataset

class CrossValidationFramework:
    def __init__(
        self,
        n_splits: int = 5,
        tree_models: Optional[Dict[str, Any]] = None,
        nn_hidden_dims: list = [512, 256, 128]
    ):
        """
        Initialize the cross-validation framework
        
        Args:
            n_splits: Number of folds for cross-validation
            tree_models: Dictionary of tree-based model instances
            nn_hidden_dims: Hidden layer dimensions for neural network
        """
        self.n_splits = n_splits
        self.nn_hidden_dims = nn_hidden_dims
        
        # Initialize tree-based models
        self.tree_models = tree_models if tree_models is not None else {
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(n_estimators=100, random_state=42)
        }
        
    def cross_validate_trees(
        self,
        real_data: np.ndarray,
        real_labels: np.ndarray,
        generated_data: Optional[np.ndarray] = None,
        generated_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, dict]]:
        """
        Perform cross-validation with tree-based models
        """
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        results = {}
        
        # Define training scenarios
        scenarios = {'real_only': (real_data, real_labels)}
        if generated_data is not None and generated_labels is not None:
            scenarios['generated_only'] = (generated_data, generated_labels)
            scenarios['mixed'] = (
                np.vstack([real_data, generated_data]),
                np.concatenate([real_labels, generated_labels])
            )
        
        for model_name, model in self.tree_models.items():
            print(f"\nCross-validating {model_name}...")
            model_results = {}
            
            for scenario_name, (X, y) in scenarios.items():
                print(f"  Scenario: {scenario_name}")
                fold_scores = []
                fold_times = []
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(real_data, real_labels)):
                    # For validation, always use real data
                    X_val, y_val = real_data[val_idx], real_labels[val_idx]
                    
                    # For training, use scenario-specific data
                    if scenario_name == 'real_only':
                        X_train = X[train_idx]
                        y_train = y[train_idx]
                    else:
                        # For generated or mixed scenarios, combine with remaining real data
                        X_train = np.vstack([X, real_data[train_idx]])
                        y_train = np.concatenate([y, real_labels[train_idx]])
                    
                    # Train and evaluate
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    # Evaluate
                    y_pred = model.predict(X_val)
                    accuracy = accuracy_score(y_val, y_pred)
                    
                    fold_scores.append(accuracy)
                    fold_times.append(train_time)
                    
                    print(f"    Fold {fold + 1}: Accuracy = {accuracy:.4f}, Time = {train_time:.2f}s")
                
                # Store results
                model_results[scenario_name] = {
                    'mean_accuracy': np.mean(fold_scores),
                    'std_accuracy': np.std(fold_scores),
                    'mean_time': np.mean(fold_times),
                    'fold_scores': fold_scores,
                    'fold_times': fold_times
                }
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        f'{model_name}/{scenario_name}/mean_cv_accuracy': np.mean(fold_scores),
                        f'{model_name}/{scenario_name}/std_cv_accuracy': np.std(fold_scores),
                        f'{model_name}/{scenario_name}/mean_cv_time': np.mean(fold_times)
                    })
            
            results[model_name] = model_results
        
        return results
    
    def cross_validate_nn(
        self,
        real_data: np.ndarray,
        real_labels: np.ndarray,
        generated_data: Optional[np.ndarray] = None,
        generated_labels: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 50,
        device: str = 'cpu'
    ) -> Dict[str, dict]:
        """
        Perform cross-validation with neural network model
        """
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        results = {}
        
        # Define training scenarios
        scenarios = {'real_only': (real_data, real_labels)}
        if generated_data is not None and generated_labels is not None:
            scenarios['generated_only'] = (generated_data, generated_labels)
            scenarios['mixed'] = (
                np.vstack([real_data, generated_data]),
                np.concatenate([real_labels, generated_labels])
            )
        
        for scenario_name, (X, y) in scenarios.items():
            print(f"\nNeural Network - Scenario: {scenario_name}")
            fold_scores = []
            fold_times = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(real_data, real_labels)):
                # For validation, always use real data
                X_val = torch.FloatTensor(real_data[val_idx]).to(device)
                y_val = torch.LongTensor(real_labels[val_idx]).to(device)
                
                # For training, use scenario-specific data
                if scenario_name == 'real_only':
                    X_train = X[train_idx]
                    y_train = y[train_idx]
                else:
                    # For generated or mixed scenarios, combine with remaining real data
                    X_train = np.vstack([X, real_data[train_idx]])
                    y_train = np.concatenate([y, real_labels[train_idx]])
                
                # Convert training data to tensors
                X_train = torch.FloatTensor(X_train).to(device)
                y_train = torch.LongTensor(y_train).to(device)
                
                # Initialize model
                model = self._create_nn_model(
                    input_dim=X_train.shape[1],
                    num_classes=len(np.unique(y_train))
                ).to(device)
                
                # Training setup
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters())
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Train model
                start_time = time.time()
                for epoch in range(epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                train_time = time.time() - start_time
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    _, predicted = torch.max(val_outputs.data, 1)
                    accuracy = (predicted == y_val).float().mean().item()
                
                fold_scores.append(accuracy)
                fold_times.append(train_time)
                
                print(f"  Fold {fold + 1}: Accuracy = {accuracy:.4f}, Time = {train_time:.2f}s")
            
            # Store results
            results[scenario_name] = {
                'mean_accuracy': np.mean(fold_scores),
                'std_accuracy': np.std(fold_scores),
                'mean_time': np.mean(fold_times),
                'fold_scores': fold_scores,
                'fold_times': fold_times
            }
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    f'neural_network/{scenario_name}/mean_cv_accuracy': np.mean(fold_scores),
                    f'neural_network/{scenario_name}/std_cv_accuracy': np.std(fold_scores),
                    f'neural_network/{scenario_name}/mean_cv_time': np.mean(fold_times)
                })
        
        return results
    
    def _create_nn_model(self, input_dim: int, num_classes: int) -> nn.Module:
        """Create neural network model"""
        layers = []
        current_dim = input_dim
        
        for h_dim in self.nn_hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, num_classes))
        
        return nn.Sequential(*layers)