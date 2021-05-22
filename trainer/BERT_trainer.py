from __future__ import absolute_import

import torch
import numpy as np

from utils.metrics import model_performance

#@title Bert training function (bert_train)
def bert_train(optimizer, 
               train_iter, 
               dev_iter, 
               model,
               loss_fn,
               device,
               number_epoch,
               model_name="approach1_model.pt",
               patience=4):
    """
    Training loop for the model, which calls on eval to evaluate after each epoch
    """
    print("Training model.")
    best_val_rmse = 10 
    patience_breach = 0 
    train_rmse = []
    valid_rmse = []

    for epoch in range(1, number_epoch+1):
        model.train()

        epoch_loss = 0
        epoch_sse = 0
        no_observations = 0  # Observations used for training so far

        for (input_id, attn_mask), target in train_iter:

            input_id = input_id.to(device, dtype=torch.long)
            attn_mask = attn_mask.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.float32)
            no_observations += target.shape[0] 
            predictions = model.forward(input_id, attn_mask)
            optimizer.zero_grad()
            target = target.unsqueeze(1)
            loss = loss_fn(predictions, target)
            sse, __ = model_performance(predictions.detach().cpu().numpy(), 
                                        target.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()*target.shape[0]
            epoch_sse += sse

        valid_loss, valid_mse, __, __ = bert_eval(dev_iter, model, device, loss_fn)
        epoch_loss, epoch_mse = epoch_loss / no_observations, epoch_sse / no_observations
        
        train_rmse.append(epoch_mse**0.5)
        valid_rmse.append(valid_mse**0.5)
        
        print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.2f} | Train MSE: {epoch_mse:.2f} | Train RMSE: {epoch_mse**0.5:.2f} | \
        Val. Loss: {valid_loss:.4f} | Val. MSE: {valid_mse:.4f} |  Val. RMSE: {valid_mse**0.5:.4f} |')

        # Early stopping
        if (valid_mse**0.5) < best_val_rmse:
          print("Found a better model, saving ... ")
          torch.save(model.state_dict(), model_name)
          patience_breach = 0
          best_val_rmse = valid_mse**0.5
        else:
          patience_breach += 1 
        if patience_breach == patience:
          print("Early stopping. Reverting back to the best model before ... ")
          return train_rmse, valid_rmse
          break 
    return train_rmse, valid_rmse  


#@title Bert eval function (bert_eval)
def bert_eval(data_iter, model, device, loss_fn):
    """
    Evaluate model performance on the dev set
    params:
      data_iter: torch dataloader
      model: model that's being trained 
    out:
      MSE and RMSE errors
    """
    model.eval()
    epoch_loss = 0
    epoch_sse = 0
    pred_all = []
    trg_all = []
    no_observations = 0

    with torch.no_grad():
        for (input_id, attn_mask), target in data_iter:
            input_id = input_id.to(device, dtype=torch.long)
            attn_mask = attn_mask.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.float32)

            predictions = model(input_id, attn_mask).squeeze(1)
            loss = loss_fn(predictions, target)
            no_observations += target.shape[0]

            # We get the mse
            pred, trg = predictions.detach().cpu().numpy(), target.detach().cpu().numpy()
            sse, __ = model_performance(pred, trg)
            epoch_loss += loss.item()*target.shape[0]
            epoch_sse += sse
            pred_all.extend(pred)
            trg_all.extend(trg)

    return epoch_loss/no_observations, epoch_sse/no_observations, \
           np.array(pred_all), np.array(trg_all)
