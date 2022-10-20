from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
from bisect import bisect_right


# 构建数据集
def split_dataset(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, shuffle=True,
                                                        random_state=2020)
    Xp_train = torch.from_numpy(np.array(X_train)).to(torch.float32)
    yp_train = torch.from_numpy(np.array(y_train)).to(torch.long)
    Xp_test = torch.from_numpy(np.array(X_test)).to(torch.float32)
    yp_test = torch.from_numpy(np.array(y_test)).to(torch.long)
    return Xp_train, yp_train, Xp_test, yp_test


class ImageDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {}  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid) in enumerate(self.data_source):
            if pid not in self.index_dic:
                self.index_dic[pid] = [index]
            else:
                self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]  # 每个pid对应的索引
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])  # 复制一份
            if len(idxs) < self.num_instances:  # 如果少于num_instances,随机选择
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)  # 打乱
            batch_idxs_dict[pid] = [idxs[i * self.num_instances: (i + 1) * self.num_instances] for i in
                                    range(len(idxs) // self.num_instances)]
        #             batch_idxs = []
        #             for idx in idxs:
        #                 batch_idxs.append(idx)
        #                 if len(batch_idxs) == self.num_instances:
        #                     batch_idxs_dict[pid].append(batch_idxs)
        #                     batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=0.1,
            warmup_iters=10,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
