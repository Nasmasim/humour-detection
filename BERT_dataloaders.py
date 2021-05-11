

# Baseline dataloader defined by joining back train and dev and splitting randomly
def get_dataloaders(input_data, 
                    targets, 
                    train_split=train_proportion):
    """
    Using outputs from 'get_input_bert', create dataloaders for training. Make 
    random splits with the training dataset. 
    """
    train_and_dev = Task1Dataset(input_data, targets)
    train_examples = round(len(train_and_dev) * train_split)
    dev_examples = len(train_and_dev) - train_examples
    # split datasets
    train_dataset, dev_dataset = random_split(train_and_dev,
                                              (train_examples,
                                                dev_examples))
    # load into torch
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               shuffle=True, 
                                              batch_size=BATCH_SIZE)
    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                             shuffle=False, 
                                             batch_size=BATCH_SIZE)

    return train_loader, dev_loader

# Data loaders for both train and validation dataset (dev or test)

def get_dataloaders_no_random_split(input_data_train, 
                                    targets_train, 
                                    input_data_valid,
                                    targets_valid):
    """
    Using outputs from 'get_input_bert', create dataloaders for training.
    """
    train_ds = Task1Dataset(input_data_train, targets_train)
    valid_ds = Task1Dataset(input_data_valid, targets_valid)
    # load into torch
    train_loader = torch.utils.data.DataLoader(train_ds, 
                                               shuffle=True, 
                                               batch_size=BATCH_SIZE)
    dev_loader = torch.utils.data.DataLoader(valid_ds, 
                                             shuffle=False,
                                             batch_size=BATCH_SIZE)

    return train_loader, dev_loader