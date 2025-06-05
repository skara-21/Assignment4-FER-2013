import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from helpers.model_analyser import analyze_model_behavior, create_learning_curves,get_connection_type,calculate_gradient_flow,analyze_connection_effectiveness,analyze_regularization_effectiveness,get_regularization_type

def train_model_depth(model, train_loader, val_loader, experiment_name, device, epochs=30):

  wandb.init(
      project="fer-2013-depth-study",
      name=experiment_name,
      config={
          "architecture": experiment_name,
          "epochs": epochs,
          "batch_size": 64,
          "learning_rate": 0.001,
          "optimizer": "Adam",
          "dataset": "FER-2013",
          "phase": "depth_study"
      },
      tags=["phase1", "depth-study", "systematic"]
  )
  
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
  
  train_losses, val_losses = [], []
  train_accs, val_accs = [], []
  best_val_acc = 0
  patience_counter = 0
  early_stopping_patience = 10
  
  total_params = sum(p.numel() for p in model.parameters())
  wandb.log({"model/total_parameters": total_params})
  
  
  for epoch in range(epochs):
      model.train()
      train_loss, train_correct, train_total = 0, 0, 0
      
      for data, target in train_loader:
          data, target = data.to(device), target.to(device)
          
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
          
          train_loss += loss.item()
          pred = output.argmax(dim=1)
          train_correct += pred.eq(target).sum().item()
          train_total += target.size(0)
      
      model.eval()
      val_loss, val_correct, val_total = 0, 0, 0
      
      with torch.no_grad():
          for data, target in val_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              loss = criterion(output, target)
              
              val_loss += loss.item()
              pred = output.argmax(dim=1)
              val_correct += pred.eq(target).sum().item()
              val_total += target.size(0)
      
      train_loss /= len(train_loader)
      val_loss /= len(val_loader)
      train_acc = 100. * train_correct / train_total
      val_acc = 100. * val_correct / val_total
      
      train_losses.append(train_loss)
      val_losses.append(val_loss)
      train_accs.append(train_acc)
      val_accs.append(val_acc)
      
      loss_gap = val_loss - train_loss
      acc_gap = train_acc - val_acc
      overfitting_score = loss_gap + max(0, acc_gap * 0.01)
      
      wandb.log({
          "epoch": epoch,
          "train_loss": train_loss,
          "val_loss": val_loss,
          "train_accuracy": train_acc,
          "val_accuracy": val_acc,
          "learning_rate": optimizer.param_groups[0]['lr'],
          "loss_gap": loss_gap,
          "accuracy_gap": acc_gap,
          "overfitting_score": overfitting_score
      })
      
      scheduler.step(val_loss)
      
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          patience_counter = 0
      else:
          patience_counter += 1
          
      if patience_counter >= early_stopping_patience:
          print(f"Early stopping at epoch {epoch+1}")
          break
          
      if epoch % 5 == 0 or epoch == epochs - 1:
          print(f'Epoch {epoch:2d}: Train: {train_acc:.1f}%/{train_loss:.3f}, '
                f'Val: {val_acc:.1f}%/{val_loss:.3f}, Gap: {loss_gap:.3f}')
  
  final_loss_gap = train_losses[-1] - val_losses[-1]
  final_acc_gap = train_accs[-1] - val_accs[-1]
  
  model_status = analyze_model_behavior(train_accs[-1], val_accs[-1], final_loss_gap)
  
  create_learning_curves(experiment_name, train_losses, val_losses, train_accs, val_accs)
  
  results = {
      'experiment_name': experiment_name,
      'best_val_accuracy': best_val_acc,
      'final_train_accuracy': train_accs[-1],
      'final_val_accuracy': val_accs[-1],
      'final_loss_gap': final_loss_gap,
      'final_acc_gap': final_acc_gap,
      'overfitting_score': final_loss_gap,
      'model_status': model_status,
      'total_epochs': len(train_losses),
      'total_parameters': total_params
  }
  
  wandb.log({f"final/{k}": v for k, v in results.items() if isinstance(v, (int, float))})
  wandb.finish()
  return results





def train_model_connections(model, train_loader, val_loader, experiment_name, device, epochs=20):

  #print('new version')
  wandb.init(
      project="fer-2013-connections-study",
      name=f"{experiment_name}",
      config={
          "architecture": experiment_name,
          "epochs": epochs,
          "batch_size": 128,
          "learning_rate": 0.002,
          "optimizer": "AdamW",
          "dataset": "FER-2013",
          "phase": "connections_study"
      },
      tags=["phase2", "skip-connections", "training"]
  )
  
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
  
  scheduler = optim.lr_scheduler.OneCycleLR(
      optimizer, 
      max_lr=0.002, 
      steps_per_epoch=len(train_loader), 
      epochs=epochs
  )
  
  train_losses, val_losses = [], []
  train_accs, val_accs = [], []
  best_val_acc = 0
  patience_counter = 0
  early_stopping_patience = 7
  
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
  wandb.log({
      "model/total_parameters": total_params,
      "model/trainable_parameters": trainable_params
  })
  
  for epoch in range(epochs):
      model.train()
      train_loss, train_correct, train_total = 0, 0, 0
      
      for data, target in train_loader:
          data, target = data.to(device), target.to(device)
          
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
          scheduler.step()
          
          train_loss += loss.item()
          pred = output.argmax(dim=1)
          train_correct += pred.eq(target).sum().item()
          train_total += target.size(0)
      
      model.eval()
      val_loss, val_correct, val_total = 0, 0, 0
      
      with torch.no_grad():
          for data, target in val_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              loss = criterion(output, target)
              
              val_loss += loss.item()
              pred = output.argmax(dim=1)
              val_correct += pred.eq(target).sum().item()
              val_total += target.size(0)
      
      train_loss /= len(train_loader)
      val_loss /= len(val_loader)
      train_acc = 100. * train_correct / train_total
      val_acc = 100. * val_correct / val_total
      
      train_losses.append(train_loss)
      val_losses.append(val_loss)
      train_accs.append(train_acc)
      val_accs.append(val_acc)
      
      loss_gap = val_loss - train_loss
      acc_gap = train_acc - val_acc
      overfitting_score = loss_gap + max(0, acc_gap * 0.01)
      
      if epoch % 3 == 0 or epoch == epochs - 1:
          wandb.log({
              "epoch": epoch,
              "train_loss": train_loss,
              "val_loss": val_loss,
              "train_accuracy": train_acc,
              "val_accuracy": val_acc,
              "learning_rate": optimizer.param_groups[0]['lr'],
              "loss_gap": loss_gap,
              "accuracy_gap": acc_gap,
              "overfitting_score": overfitting_score
          })
      
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          patience_counter = 0
      else:
          patience_counter += 1
          
      if patience_counter >= early_stopping_patience:
          print(f"Early stopping at epoch {epoch+1}")
          break
          
      if epoch % 5 == 0 or epoch == epochs - 1:
          print(f'Epoch {epoch:2d}: Train: {train_acc:.1f}%, Val: {val_acc:.1f}%, Gap: {loss_gap:.3f}')
  
  final_loss_gap = train_losses[-1] - val_losses[-1]
  final_acc_gap = train_accs[-1] - val_accs[-1]
  
  model_status = analyze_model_behavior(train_accs[-1], val_accs[-1], final_loss_gap)
  
  create_learning_curves(experiment_name, train_losses, val_losses, train_accs, val_accs)
  
  connection_effectiveness = analyze_connection_effectiveness(
      experiment_name, final_loss_gap, final_acc_gap
  )
  
  results = {
      'experiment_name': experiment_name,
      'connection_type': get_connection_type(experiment_name),
      'best_val_accuracy': best_val_acc,
      'final_train_accuracy': train_accs[-1],
      'final_val_accuracy': val_accs[-1],
      'final_loss_gap': final_loss_gap,
      'final_acc_gap': final_acc_gap,
      'overfitting_score': final_loss_gap,
      'model_status': model_status,
      'connection_effectiveness': connection_effectiveness,
      'total_epochs': len(train_losses),
      'total_parameters': total_params
  }
  
  numeric_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}
  wandb.log({f"final/{k}": v for k, v in numeric_results.items()})
  
  wandb.log({
      "info/experiment_name": experiment_name,
      "info/connection_type": get_connection_type(experiment_name),
      "info/model_status": model_status,
      "info/connection_effectiveness": connection_effectiveness
  })
  wandb.finish()
  return results



def train_model_regularization(model, train_loader, val_loader, experiment_name, device, epochs=18):
  wandb.init(
    project="fer-2013-regularization-study",
    name=experiment_name,
    config={
        "architecture": experiment_name,
        "epochs": epochs,
        "batch_size": 64,
        "learning_rate": 0.002,
        "optimizer": "AdamW",
        "dataset": "FER-2013",
        "phase": "regularization_study"
    },
    tags=["phase3", "regularization", "systematic"]
  )

  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
  
  scheduler = optim.lr_scheduler.OneCycleLR(
      optimizer, 
      max_lr=0.002, 
      steps_per_epoch=len(train_loader), 
      epochs=epochs
  )

  train_losses, val_losses = [], []
  train_accs, val_accs = [], []
  best_val_acc = 0
  patience_counter = 0
  early_stopping_patience = 7

  total_params = sum(p.numel() for p in model.parameters())
  wandb.log({"model/total_parameters": total_params})

  for epoch in range(epochs):
      model.train()
      train_loss, train_correct, train_total = 0, 0, 0
      
      for data, target in train_loader:
          data, target = data.to(device), target.to(device)
          
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
          scheduler.step()
          
          train_loss += loss.item()
          pred = output.argmax(dim=1)
          train_correct += pred.eq(target).sum().item()
          train_total += target.size(0)
      
      model.eval()
      val_loss, val_correct, val_total = 0, 0, 0
      
      with torch.no_grad():
          for data, target in val_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              loss = criterion(output, target)
              
              val_loss += loss.item()
              pred = output.argmax(dim=1)
              val_correct += pred.eq(target).sum().item()
              val_total += target.size(0)
      
      train_loss /= len(train_loader)
      val_loss /= len(val_loader)
      train_acc = 100. * train_correct / train_total
      val_acc = 100. * val_correct / val_total
      
      train_losses.append(train_loss)
      val_losses.append(val_loss)
      train_accs.append(train_acc)
      val_accs.append(val_acc)
      
      loss_gap = val_loss - train_loss
      acc_gap = train_acc - val_acc
      overfitting_score = loss_gap + max(0, acc_gap * 0.01)
      
      if epoch % 3 == 0 or epoch == epochs - 1:
          wandb.log({
              "epoch": epoch,
              "train_loss": train_loss,
              "val_loss": val_loss,
              "train_accuracy": train_acc,
              "val_accuracy": val_acc,
              "learning_rate": optimizer.param_groups[0]['lr'],
              "loss_gap": loss_gap,
              "accuracy_gap": acc_gap,
              "overfitting_score": overfitting_score
          })
      
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          patience_counter = 0
      else:
          patience_counter += 1
          
      if patience_counter >= early_stopping_patience:
          print(f"Early stopping at epoch {epoch+1}")
          break
          
      if epoch % 5 == 0 or epoch == epochs - 1:
          print(f'Epoch {epoch:2d}: Train: {train_acc:.1f}%/{train_loss:.3f}, '
                f'Val: {val_acc:.1f}%/{val_loss:.3f}, Gap: {loss_gap:.3f}')

  final_loss_gap = train_losses[-1] - val_losses[-1]
  final_acc_gap = train_accs[-1] - val_accs[-1]

  model_status = analyze_model_behavior(train_accs[-1], val_accs[-1], final_loss_gap)
  create_learning_curves(experiment_name, train_losses, val_losses, train_accs, val_accs)

  regularization_effectiveness = analyze_regularization_effectiveness(
      experiment_name, final_loss_gap, final_acc_gap
  )

  results = {
      'experiment_name': experiment_name,
      'regularization_type': get_regularization_type(experiment_name),
      'best_val_accuracy': best_val_acc,
      'final_train_accuracy': train_accs[-1],
      'final_val_accuracy': val_accs[-1],
      'final_loss_gap': final_loss_gap,
      'final_acc_gap': final_acc_gap,
      'overfitting_score': final_loss_gap,
      'model_status': model_status,
      'regularization_effectiveness': regularization_effectiveness,
      'total_epochs': len(train_losses),
      'total_parameters': total_params
  }

  numeric_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}
  wandb.log({f"final/{k}": v for k, v in numeric_results.items()})

  wandb.log({
      "info/experiment_name": experiment_name,
      "info/regularization_type": get_regularization_type(experiment_name),
      "info/model_status": model_status,
      "info/regularization_effectiveness": regularization_effectiveness
  })  
  wandb.finish()
  return results
