# 时间步信息需要单独处理：时间步是类别变量，不宜与连续特征统一缩放
time_steps = node_features[:, 0].copy()  # 第1个特征是时间步

# 对连续特征（特征2-166）进行标准化
continuous_features = node_features[:, 1:]  # 注意：实际索引需根据特征分布调整
scaler = StandardScaler()
continuous_features_scaled = scaler.fit_transform(continuous_features)

# 对时间步进行独热编码（可选）或保留原始值
# 方案1：保留原始时间步并做单独标准化
time_step_scaled = (time_steps - 1) / 48  # 归一化到[0,1]

# 方案2：独热编码（推荐，因为时间步是类别性质的）
from sklearn.preprocessing import OneHotEncoder
time_onehot = OneHotEncoder(sparse_output=False).fit_transform(time_steps.reshape(-1, 1))

# 组合特征
final_features = np.hstack([time_onehot, continuous_features_scaled])
print(f"最终特征维度: {final_features.shape[1]}")  # 49(时间步独热) + 165 = 214维