from torch.utils.data import WeightedRandomSampler
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# 方法一：样本权重法（用于训练时加权损失）
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),  # 合法=0，非法=1
    y=labels[labels != -1]  # 排除未知节点
)
class_weight_tensor = torch.tensor(class_weights, dtype=torch.float)

# 方法二：对已知标签节点进行重采样
labeled_indices = np.where(labels != -1)[0]
licit_indices = np.where(labels == 0)[0]
illicit_indices = np.where(labels == 1)[0]

# 对非法类别进行过采样（复制样本）
oversampled_illicit = np.random.choice(
    illicit_indices, 
    size=len(licit_indices), 
    replace=True
)
balanced_indices = np.concatenate([licit_indices, oversampled_illicit])
np.random.shuffle(balanced_indices)

# 方法三：对邻居采样时优先采样非法节点的邻居（GraphSAGE特有优势）
# 详见采样策略部分