import numpy as np
import torch
import random
def init_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('Inputs to shuffles must be have the same length')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
def mini_batch(*arrays, **kwargs):
    batch_size = kwargs.get('batch_size', 2048)

    if len(arrays) == 1:
        for i in range(0, len(arrays[0]), batch_size):
            yield arrays[0][i: i+batch_size]
    else:
        for i in range(0, len(arrays[0]), batch_size):
            yield tuple(array[i: i+batch_size] for array in arrays)
            
def cri_mini_batch(data_list, batch_size=2048):

    num_data = len(data_list)

    indices = list(range(num_data))
    random.shuffle(indices)


    for i in range(0, num_data, batch_size):

        batch_indices = indices[i:i + batch_size]

        batch = [data_list[j] for j in batch_indices]

        users = [item[0] for item in batch]
        pos_items = [item[1] for item in batch]
        bpr_neg_items = [item[2] for item in batch]
        cl_neg_items = [item[3] for item in batch]

        yield torch.LongTensor(users), \
              torch.LongTensor(pos_items), \
              torch.LongTensor(bpr_neg_items), \
              torch.LongTensor(cl_neg_items)


def convert_sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    value = torch.FloatTensor(coo.data)
    sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
    sp_tensor = sp_tensor.coalesce()
    return sp_tensor