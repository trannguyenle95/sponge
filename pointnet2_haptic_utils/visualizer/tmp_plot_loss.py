import numpy as np
from matplotlib import pyplot as plt

log_path = 'log/sem_seg/2021-10-13_00-34/logs/pointnet2_sem_seg_msg.txt'
f = open(log_path, 'r')
lines = f.readlines()
f.close()

train_losses = []
test_losses = []
train_accs = []
test_accs = []
for line in lines:
    strs = [
        'Training mean loss: ',
        'Training accuracy: ',
        'eval mean loss: ',
        'eval point accuracy: '
    ]

    lists = [train_losses, train_accs, test_losses, test_accs]
    for s, l in zip(strs, lists):
        idx = line.find(s)
        if idx != -1:
            item = float(line[idx + len(s):])
            l.append(item)
            break

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes = axes.reshape(-1)
axes[0].plot(range(len(train_accs)), train_accs, label='train')
axes[0].plot([i for i in range(len(test_accs))], test_accs, label='test')
axes[1].plot(range(len(train_losses)), train_losses, label='train')
axes[1].plot([i for i in range(len(test_losses))], test_losses, label='test')
axes[0].set_title("accuracy")
axes[1].set_title("loss")
plt.legend()
plt.show()

