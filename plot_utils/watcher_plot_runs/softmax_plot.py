import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import softmax
from torch.nn import functional as F
from torch import Tensor

def softmax_with_torch(x, temperature):
    return F.softmax(Tensor(x/temperature), dim=0).numpy()

for temperature in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    n_states = 100
    n_seeds = 100
    average_probs = np.zeros(n_states)

    for i in range(n_seeds):
        np.random.seed(i)
        # Generate uniformly distributed variable between 0 and 1
        random_values = np.random.rand(n_states)
        after_softmax = softmax_with_torch(random_values, temperature)
        after_softmax_sorted = np.sort(after_softmax)
        average_probs += after_softmax_sorted
    average_probs /= n_seeds

    indices = np.arange(len(average_probs)).reshape(-1)
    plt.bar(indices, average_probs)
    # plt.title("Probability")
    plt.xlabel("State", fontsize=20)
    plt.ylabel("Probability", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    save_folder_path = '../../figures/softmax'
    save_name = 'softmax_temp%s.png' % str(temperature)
    save_path = os.path.join(save_folder_path, save_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()