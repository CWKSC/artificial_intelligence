

import math
import torch
import seaborn as sns
import matplotlib.pyplot as plt

words = 32

T = torch.nn.Transformer(num_encoder_layers=3)
seq = torch.rand(words, 512)
print(seq.shape)
out = T(seq, seq)

with torch.no_grad():
    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(11,5), gridspec_kw={'width_ratios':[.5,.5,0.05]})
    map1 = seq @ seq.T
    sns.heatmap(map1.numpy(),cmap="YlGnBu",cbar=False,ax=ax1)
    map2 = seq @ out.T
    sns.heatmap(map2.numpy(),cmap="YlGnBu",ax=ax2,cbar_ax=ax3)

plt.show()