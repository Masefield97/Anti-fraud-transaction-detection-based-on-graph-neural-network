def evaluate_temporal_generalization(model, data, time_steps):
    """评估模型在不同时间步上的表现，验证归纳泛化能力"""
    model.eval()
    device = next(model.parameters()).device
    
    results = {}
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        logits = F.softmax(out, dim=1)[:, 1]  # 非法类别概率
        
        for ts in range(1, 50):
            ts_mask = (time_steps == ts) & (data.y.cpu().numpy() != -1)
            if ts_mask.sum() == 0:
                continue
            
            preds = (logits[ts_mask] > 0.5).cpu().numpy()
            labels = data.y[ts_mask].cpu().numpy()
            
            if len(np.unique(labels)) > 1:
                f1 = f1_score(labels, preds, average='binary')
            else:
                f1 = 0 if labels[0] == 1 else 1
            
            results[ts] = f1
    
    return results

# 可视化时间泛化曲线
import matplotlib.pyplot as plt

time_step_f1 = evaluate_temporal_generalization(model, data, time_steps_array)
ts_list = sorted(time_step_f1.keys())
f1_list = [time_step_f1[ts] for ts in ts_list]

plt.figure(figsize=(12, 5))
plt.plot(ts_list, f1_list, marker='o', linewidth=2)
plt.axvline(x=35.5, color='r', linestyle='--', label='训练/验证划分')
plt.axvline(x=42.5, color='orange', linestyle='--', label='验证/测试划分')
plt.xlabel('Time Step')
plt.ylabel('F1 Score')
plt.title('Temporal Generalization of GraphSAGE on Elliptic Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()