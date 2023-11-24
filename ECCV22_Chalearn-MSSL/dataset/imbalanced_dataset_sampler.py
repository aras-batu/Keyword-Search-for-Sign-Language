import torch
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)
            
        label_to_count = {}
        for idx in self.indices:
            label = dataset.df.iloc[idx]['label']
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        print(label_to_count)
        weights = [1.0 / label_to_count[dataset.df.iloc[idx]['label']]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
                
    def __iter__(self):
        # TODO: ADD NEGATIVE SAMPLING IN HERE
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples