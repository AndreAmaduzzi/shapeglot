import numpy as np
from torch.utils.data import Dataset

def shuffle_ids(ids, label, random_seed=None):
    ''' e.g. if [a, b] with label 0 makes it [b, a] with label 1.
    '''
    res_ids = ids.copy()   # initialization of output list
    if random_seed is not None:
        np.random.seed(random_seed)
    shuffle = np.random.shuffle
    idx = [0, 1]
    shuffle(idx)
    # if idx==0, no swap
    # if idx==1, swap elements of ids
    i=0
    for one_idx in idx:
        res_ids[i] = ids[one_idx]
        i += 1 

    target = idx[0]

    return res_ids, target


class ShapeglotDataset(Dataset):
    def __init__(self, np_data, shuffle_geo=False, target_last=False):
        """
        :param np_data:
        :param shuffle_geo: if True, the positions of the shapes in context are randomly swapped.
        """
        super(ShapeglotDataset, self).__init__()
        self.data = np_data
        self.shuffle_geo = shuffle_geo
        self.target_last = target_last

    def __getitem__(self, index):
        text = self.data['text'][index].astype(np.long)
        geos = self.data['in_geo'][index].astype(np.long)
        target = self.data['target'][index]
        idx = np.arange(len(geos))

        if self.shuffle_geo:
            np.random.shuffle(idx)

        geos = geos[idx]
        target = np.argmax(target[idx])

        if self.target_last:
            last = geos[-1]
            geos[-1] = geos[target]
            geos[target] = last
            target = len(geos) - 1

        return geos, target, text

    def __len__(self):
        return len(self.data)
