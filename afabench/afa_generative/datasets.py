from torch.utils.data import DataLoader


def prepare_datasets(train_dataset, val_dataset, batch_size: int):
    # Get dimensions using shape properties
    d_in = train_dataset.feature_shape[0]
    d_out = train_dataset.label_shape[0]

    # Create new datasets with converted data format
    class ConvertedDataset:
        def __init__(self, original_dataset):
            self.original_dataset = original_dataset
            self.features, self.labels = original_dataset.get_all_data()
            self.features = self.features.float()
            self.labels = self.labels.argmax(dim=1).long()

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

        def __len__(self):
            return len(self.original_dataset)

    train_dataset = ConvertedDataset(train_dataset)
    val_dataset = ConvertedDataset(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, pin_memory=True
    )

    return train_loader, val_loader, d_in, d_out
