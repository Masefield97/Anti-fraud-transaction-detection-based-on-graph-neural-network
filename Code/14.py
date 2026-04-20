import shap

def explain_predictions(model, data, sample_indices):
    """使用SHAP解释模型预测"""
    device = next(model.parameters()).device
    
    # 定义预测函数
    def predict_proba(x_batch):
        model.eval()
        with torch.no_grad():
            # 简化版：由于SHAP需要独立输入，这里做近似解释
            out = model(data.x.to(device), data.edge_index.to(device))
            return F.softmax(out, dim=1).cpu().numpy()
    
    # 对采样节点进行解释
    sample_features = data.x[sample_indices].cpu().numpy()
    explainer = shap.Explainer(predict_proba, sample_features)
    shap_values = explainer(sample_features)
    
    # 可视化
    shap.summary_plot(shap_values[:, :, 1], sample_features, 
                      feature_names=[f'f{i}' for i in range(sample_features.shape[1])])