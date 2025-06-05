import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.optim as optim

#რიცხვები არის ამ კონკრეტული მონაცემების მიხედვით აღებული
def analyze_model_behavior(train_acc, val_acc, loss_gap):
  if train_acc < 25:
      return "UNDERFITTING"
  elif loss_gap > 0.5 or (train_acc - val_acc) > 20:
      return "OVERFITTING"
  else:
      return "GOOD_BALANCE"

#ეს მე არ დამიწერია ნამვილად
def create_learning_curves(experiment_name, train_losses, val_losses, train_accs, val_accs):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  
  epochs_range = range(len(train_losses))
  
  ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
  ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
  ax1.set_title(f'{experiment_name} - Loss Curves')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  ax1.legend()
  ax1.grid(True, alpha=0.3)
  
  ax2.plot(epochs_range, train_accs, 'b-', label='Training Accuracy', linewidth=2)
  ax2.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
  ax2.set_title(f'{experiment_name} - Accuracy Curves')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy (%)')
  ax2.legend()
  ax2.grid(True, alpha=0.3)
  
  plt.tight_layout()
  wandb.log({"learning_curves": wandb.Image(fig)})
  plt.show()

def get_connection_type(experiment_name):
  if 'standard' in experiment_name.lower():
      return "Standard"
  elif 'skip' in experiment_name.lower():
      return "Skip Connections"
  elif 'hybrid' in experiment_name.lower():
      return "Hybrid Skip Connections"
  elif 'dense' in experiment_name.lower():
      return "Dense Connections"
  else:
      return "Unknown"


def calculate_gradient_flow(model):
  total_norm = 0
  param_count = 0
  
  for p in model.parameters():
      if p.grad is not None:
          param_norm = p.grad.data.norm(2)
          total_norm += param_norm.item() ** 2
          param_count += 1
  
  if param_count == 0:
      return 0.0
      
  avg_grad_norm = (total_norm ** 0.5) / param_count
  return avg_grad_norm

#რიცხვები არის ამ კონკრეტული მონაცემების მიხედვით აღებული
def analyze_connection_effectiveness(experiment_name, loss_gap, acc_gap):
  connection_type = get_connection_type(experiment_name)
  
  if connection_type == "Standard":
      return "Baseline"
  
  if loss_gap < 0.3 and acc_gap < 5:
      return "Highly Effective"
  elif loss_gap < 0.6 and acc_gap < 12:
      return "Moderately Effective"
  elif loss_gap < 1.0 and acc_gap < 18:
      return "Slightly Effective"
  else:
      return "Not Effective"

#რიცხვები არის ამ კონკრეტული მონაცემების მიხედვით აღებული
def get_regularization_type(experiment_name):
  if 'no-reg' in experiment_name:
      return "No Regularization"
  elif 'dropout' in experiment_name and 'both' not in experiment_name:
      return "Dropout Only"
  elif 'batchnorm' in experiment_name and 'both' not in experiment_name:
      return "Batch Normalization Only"
  elif 'both' in experiment_name:
      return "Both Dropout + BatchNorm"
  else:
      return "Unknown"

#რიცხვები არის ამ კონკრეტული მონაცემების მიხედვით აღებული
def analyze_regularization_effectiveness(experiment_name, loss_gap, acc_gap):
  reg_type = get_regularization_type(experiment_name)
  
  if reg_type == "No Regularization":
      return "Baseline"
  
  if loss_gap < 0.2 and acc_gap < 3:
      return "Highly Effective"
  elif loss_gap < 0.5 and acc_gap < 8:
      return "Moderately Effective"
  elif loss_gap < 0.8 and acc_gap < 15:
      return "Slightly Effective"
  else:
      return "Not Effective"
