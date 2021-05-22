# from utils_metrics import model_performance

# from torch.utils.data import Dataset
# import torch
# import numpy as np

# #@title Dataset definition
# # class Task1Dataset(Dataset):

# #     def __init__(self, train_data, labels):
# #         self.x_train = train_data
# #         self.y_train = labels

# #     def __len__(self):
# #         return len(self.y_train)

# #     def __getitem__(self, item):
# #         return self.x_train[item], self.y_train[item]

# #@title RNN training function (train)
# # We define our training loop
# def _train(train_iter, dev_iter, model, number_epoch, device, optimizer, loss_fn):
#     """
#     Training loop for the model, which calls on eval to evaluate after each epoch
#     """
    
#     print("Training model.")

#     for epoch in range(1, number_epoch+1):

#         model.train()
#         epoch_loss = 0
#         epoch_sse = 0
#         no_observations = 0  # Observations used for training so far

#         for batch in train_iter:

#             feature, target = batch

#             feature, target = feature.to(device), target.to(device)

#             # for RNN:
#             model.batch_size = target.shape[0]
#             no_observations = no_observations + target.shape[0]
#             model.hidden = model.init_hidden()

#             predictions = model(feature).squeeze(1)

#             optimizer.zero_grad()

#             loss = loss_fn(predictions, target)

#             sse, __ = model_performance(predictions.detach().cpu().numpy(), target.detach().cpu().numpy())

#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()*target.shape[0]
#             epoch_sse += sse

#         valid_loss, valid_mse, __, __ = _eval(dev_iter, model, device, loss_fn)

#         epoch_loss, epoch_mse = epoch_loss / no_observations, epoch_sse / no_observations
#         print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.2f} | Train MSE: {epoch_mse:.2f} | Train RMSE: {epoch_mse**0.5:.2f} | \
#         Val. Loss: {valid_loss:.2f} | Val. MSE: {valid_mse:.2f} |  Val. RMSE: {valid_mse**0.5:.2f} |')
        
# #@title RNN eval function (eval)
# # We evaluate performance on our dev set
# def _eval(data_iter, model, device, loss_fn):
#     """
#     Evaluating model performance on the dev set
#     """
#     model.eval()
#     epoch_loss = 0
#     epoch_sse = 0
#     pred_all = []
#     trg_all = []
#     no_observations = 0

#     with torch.no_grad():
#         for batch in data_iter:
#             feature, target = batch

#             feature, target = feature.to(device), target.to(device)

#             # for RNN:
#             model.batch_size = target.shape[0]
#             no_observations = no_observations + target.shape[0]
#             model.hidden = model.init_hidden()

#             predictions = model(feature).squeeze(1)
#             loss = loss_fn(predictions, target)

#             # We get the mse
#             pred, trg = predictions.detach().cpu().numpy(), target.detach().cpu().numpy()
#             sse, __ = model_performance(pred, trg)

#             epoch_loss += loss.item()*target.shape[0]
#             epoch_sse += sse
#             pred_all.extend(pred)
#             trg_all.extend(trg)

#         return epoch_loss/no_observations, epoch_sse/no_observations, np.array(pred_all), np.array(trg_all)