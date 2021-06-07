def get_{{customize_name}}_dataloader(dataset_path, input_size, batch_size, num_workers, train_portion=1):
    """ Prepare dataset for training and evaluating pipeline

    Args:
        dataset_path (str)
        input_size (int)
        batch_size (int)
        num_workers (int)
        train_portion (float)

    Return:
        train_loader (torch.utils.data.DataLoader)
        val_loader (torch.utils.data.DataLoader)
        test_loader (torch.utils.data.DataLoader)
    """
    # Write your code here
    return train_loader, val_loader, test_loader
