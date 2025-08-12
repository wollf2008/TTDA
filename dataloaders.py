from torch.utils.data import DataLoader


class InfiniteDataLoader(DataLoader):

    def __init__(self, *args, shuffle=True, **kwargs):
        super(InfiniteDataLoader, self).__init__(*args, **kwargs)
        self.dataset_iterator = super(InfiniteDataLoader, self).__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
