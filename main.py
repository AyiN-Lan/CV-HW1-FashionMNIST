import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimSun','Times New Roman']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family']='sans-serif'
from sklearn.metrics import confusion_matrix,accuracy_score
from torchvision.datasets import FashionMNIST

LABEL_MAP={0:"T恤/上衣",1:"裤子",2:"套头衫",3:"连衣裙",4:"外套",5:"凉鞋",6:"衬衫",7:"运动鞋",8:"包",9:"短靴"}
np.random.seed(42)
EPOCHS=50
BATCH_SIZE=64
LR_DECAY=0.9

def load_data():
    os.environ['TORCH_HOME']='./datasets'
    os.environ['TORCHVISION_DATASET_DOWNLOAD_MIRROR']='https://mirrors.tuna.tsinghua.edu.cn/pytorch/'
    train_set=FashionMNIST(root="./datasets",train=True,download=True)
    test_set=FashionMNIST(root="./datasets",train=False,download=True)
    X_train=train_set.data.numpy().astype(np.float32)/255.0
    y_train=train_set.targets.numpy()
    X_test=test_set.data.numpy().astype(np.float32)/255.0
    y_test=test_set.targets.numpy()
    X_train=X_train.reshape(-1,784)
    X_test=X_test.reshape(-1,784)
    def one_hot(y):
        return np.eye(10)[y]
    y_train_oh=one_hot(y_train)
    y_test_oh=one_hot(y_test)
    idx=np.random.permutation(len(X_train))
    split=int(0.8*len(X_train))
    X_tr,X_val=X_train[idx[:split]],X_train[idx[split:]]
    y_tr,y_val=y_train_oh[idx[:split]],y_train_oh[idx[split:]]
    return X_tr,y_tr,X_val,y_val,X_test,y_test,y_test_oh

def relu(x):
    return np.maximum(0,x)
def relu_deriv(x):
    return (x>0).astype(np.float32)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_deriv(x):
    s=sigmoid(x)
    return s*(1-s)
def softmax(x):
    exps=np.exp(x-np.max(x,axis=1,keepdims=True))
    return exps/np.sum(exps,axis=1,keepdims=True)
def cross_entropy_loss(y_pred,y_true,weights,lambda_l2):
    ce=-np.mean(np.sum(y_true*np.log(y_pred+1e-8),axis=1))
    l2=0.5*lambda_l2*sum([np.sum(w**2) for w in weights])
    return ce+l2

class MLP:
    def __init__(self,hidden_dim=128,activation="relu"):
        if activation=="relu":
            self.act=relu
            self.act_d=relu_deriv
        elif activation=="sigmoid":
            self.act=sigmoid
            self.act_d=sigmoid_deriv
        self.W1=np.random.randn(784,hidden_dim)/784
        self.b1=np.zeros((1,hidden_dim))
        self.W2=np.random.randn(hidden_dim,hidden_dim)/hidden_dim
        self.b2=np.zeros((1,hidden_dim))
        self.W3=np.random.randn(hidden_dim,10)/hidden_dim
        self.b3=np.zeros((1,10))
        self.cache={}
    def forward(self,X):
        Z1=X@self.W1+self.b1
        A1=self.act(Z1)
        Z2=A1@self.W2+self.b2
        A2=self.act(Z2)
        Z3=A2@self.W3+self.b3
        A3=softmax(Z3)
        self.cache={"X":X,"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2,"Z3":Z3,"A3":A3}
    def backward(self,y_true,lambda_l2):
        m=len(y_true)
        dZ3=self.cache["A3"]-y_true
        dW3=(self.cache["A2"].T@dZ3)/m+lambda_l2*self.W3
        db3=np.sum(dZ3,0,keepdims=True)/m
        dA2=dZ3@self.W3.T
        dZ2=dA2*self.act_d(self.cache["Z2"])
        dW2=(self.cache["A1"].T@dZ2)/m+lambda_l2*self.W2
        db2=np.sum(dZ2,0,keepdims=True)/m
        dA1=dZ2@self.W2.T
        dZ1=dA1*self.act_d(self.cache["Z1"])
        dW1=(self.cache["X"].T@dZ1)/m+lambda_l2*self.W1
        db1=np.sum(dZ1,0,keepdims=True)/m
        return {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2,"dW3":dW3,"db3":db3}
    def sgd_step(self,grads,lr):
        self.W1-=lr*grads["dW1"]
        self.b1-=lr*grads["db1"]
        self.W2-=lr*grads["dW2"]
        self.b2-=lr*grads["db2"]
        self.W3-=lr*grads["dW3"]
        self.b3-=lr*grads["db3"]
    def save_weights(self,path="./best_model.npz"):
        np.savez(path,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,W3=self.W3,b3=self.b3)
    def load_weights(self,path="./best_model.npz"):
        w=np.load(path)
        self.W1=w["W1"]
        self.b1=w["b1"]
        self.W2=w["W2"]
        self.b2=w["b2"]
        self.W3=w["W3"]
        self.b3=w["b3"]

def train_single_model(model,X_tr,y_tr,X_val,y_val,epochs,batch_size,init_lr,lr_decay,lambda_l2,save_path):
    train_loss=[]
    val_loss=[]
    val_acc=[]
    best_acc=0.0
    current_lr=init_lr
    for epoch in range(epochs):
        epoch_loss=0
        indices=np.random.permutation(len(X_tr))
        X_shuffled=X_tr[indices]
        y_shuffled=y_tr[indices]
        for i in range(0,len(X_tr),batch_size):
            Xb=X_shuffled[i:i+batch_size]
            yb=y_shuffled[i:i+batch_size]
            model.forward(Xb)
            loss=cross_entropy_loss(model.cache["A3"],yb,[model.W1,model.W2,model.W3],lambda_l2)
            epoch_loss+=loss*len(Xb)
            grads=model.backward(yb,lambda_l2)
            model.sgd_step(grads,current_lr)
        epoch_loss/=len(X_tr)
        train_loss.append(epoch_loss)
        model.forward(X_val)
        v_loss=cross_entropy_loss(model.cache["A3"],y_val,[model.W1,model.W2,model.W3],lambda_l2)
        val_loss.append(v_loss)
        acc=accuracy_score(np.argmax(y_val,axis=1),np.argmax(model.cache["A3"],axis=1))
        val_acc.append(acc)
        if acc>best_acc:
            best_acc=acc
            model.save_weights(save_path)
        current_lr*=lr_decay
        if (epoch+1)%5==0:
            print(f"Epoch {epoch+1}/{epochs} 训练损失: {epoch_loss:.4f} 验证准确率: {acc:.4f}")
    return {"train_loss":train_loss,"val_loss":val_loss,"val_acc":val_acc,"best_val_acc":best_acc}

def grid_search(X_tr,y_tr,X_val,y_val,search_space,epochs=EPOCHS,batch_size=BATCH_SIZE):
    best_params=None
    best_acc=0.0
    results=[]
    print("\n超参数网格搜索  ")
    print(f"搜索空间: {search_space}")
    for hd in search_space["hidden_dim"]:
        for lr in search_space["learning_rate"]:
            for l2 in search_space["lambda_l2"]:
                print(f"\n当前参数: 隐藏层={hd}, 学习率={lr}, L2={l2}")
                model=MLP(hidden_dim=hd,activation="relu")
                hist=train_single_model(model,X_tr,y_tr,X_val,y_val,epochs,batch_size,lr,0.9,l2,"./tmp.npz")
                current_acc=hist["best_val_acc"]
                results.append({"hidden_dim":hd,"learning_rate":lr,"lambda_l2":l2,"best_acc":current_acc})
                if current_acc>best_acc:
                    best_acc=current_acc
                    best_params={"hidden_dim":hd,"learning_rate":lr,"lambda_l2":l2}
                    print(f"新最优模型 验证准确率: {best_acc:.4f}")
    print("\n超参数搜索完成  ")
    print(f"最优参数: {best_params}")
    print(f"最优验证准确率: {best_acc:.4f}")
    return best_params,results

def evaluate_and_visualize(model,X_test,y_test,y_test_oh,train_history):
    model.forward(X_test)
    y_pred=np.argmax(model.cache["A3"],axis=1)
    test_acc=accuracy_score(y_test,y_pred)
    print(f"\n测试集最终准确率: {test_acc:.4f}")
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(train_history["train_loss"],label="训练集Loss")
    plt.plot(train_history["val_loss"],label="验证集Loss")
    plt.title("损失曲线")
    plt.xlabel("轮次")
    plt.ylabel("损失值")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_history["val_acc"],label="验证集",color="orange")
    plt.title("准确率曲线")
    plt.xlabel("轮次")
    plt.ylabel("准确率")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./train_curve.png",dpi=300,bbox_inches="tight")
    plt.show()
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(10,8))
    plt.imshow(cm,cmap="Blues")
    plt.title("混淆矩阵")
    plt.colorbar()
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.xticks(range(10),LABEL_MAP.values(),rotation=45)
    plt.yticks(range(10),LABEL_MAP.values())
    for i in range(10):
        for j in range(10):
            plt.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>500 else "black")
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png",dpi=300,bbox_inches="tight")
    plt.show()
    W1=model.W1
    hd=W1.shape[1]
    cols=16
    rows=int(np.ceil(hd/cols))
    plt.figure(figsize=(cols*1.5,rows*1.5))
    for i in range(hd):
        plt.subplot(rows,cols,i+1)
        plt.imshow(W1[:,i].reshape(28,28),cmap="gray")
        plt.title(f"{i+1}",fontsize=7)
        plt.axis("off")
    plt.suptitle(f"隐藏神经元:{hd}")
    plt.tight_layout()
    plt.savefig("./all_neurons.png",dpi=300,bbox_inches="tight")
    plt.show()
    error_idx=np.where(y_pred!=y_test)[0]
    with open("error_log.txt","w",encoding="utf-8") as f:
        f.write(f"错误数:{len(error_idx)}\n")
        for i,idx in enumerate(error_idx):
            f.write(f"{i+1} 序号:{idx} 真实:{LABEL_MAP[y_test[idx]]} 预测:{LABEL_MAP[y_pred[idx]]}\n")
    print("错例已保存到 error_log.txt")
    selected_idx=np.random.choice(error_idx,8,replace=False) if len(error_idx)>=8 else error_idx
    plt.figure(figsize=(16,8))
    for i,idx in enumerate(selected_idx):
        plt.subplot(2,4,i+1)
        plt.imshow(X_test[idx].reshape(28,28),cmap="gray")
        plt.title(f"{idx}\n{LABEL_MAP[y_test[idx]]}\n{LABEL_MAP[y_pred[idx]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("./error_analysis.png",dpi=300,bbox_inches="tight")
    plt.show()

if __name__=="__main__":
    X_tr,y_tr,X_val,y_val,X_test,y_test,y_test_oh=load_data()
    print("数据加载完成")
    SEARCH_SPACE={"hidden_dim":[64,128,256],"learning_rate":[0.1,0.01,0.001],"lambda_l2":[1e-5,1e-4,1e-3]}
    best_params,search_results=grid_search(X_tr,y_tr,X_val,y_val,SEARCH_SPACE,epochs=EPOCHS,batch_size=BATCH_SIZE)
    final_model=MLP(hidden_dim=best_params["hidden_dim"],activation="relu")
    train_history=train_single_model(final_model,X_tr,y_tr,X_val,y_val,EPOCHS,BATCH_SIZE,best_params["learning_rate"],LR_DECAY,best_params["lambda_l2"],"./final_best_model.npz")
    final_model.load_weights("./final_best_model.npz")
    evaluate_and_visualize(final_model,X_test,y_test,y_test_oh,train_history)
