import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun','Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

weights=np.load("final_best_model.npz")
W1=weights["W1"]

per_image=64
rows=8
cols=8

for part in range(4):
    start=part*per_image
    end=start+per_image
    indices=range(start,end)
    
    plt.figure(figsize=(cols*1.8,rows*1.8))
    for i,idx in enumerate(indices):
        plt.subplot(rows,cols,i+1)
        neuron_img=W1[:,idx].reshape(28,28)
        plt.imshow(neuron_img,cmap="gray")
        plt.axis("off")
    
    plt.suptitle(f"第一层神经元 第{part+1}组（{start+1}–{end}号）",fontsize=14)
    plt.tight_layout()
    plt.savefig(f"neurons_part{part+1}.png",dpi=300,bbox_inches="tight")
    plt.close()
    print(f"已生成：neurons_part{part+1}.png（第{start+1}-{end}号神经元）")

print("\n完成")
