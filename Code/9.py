import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        out = model(batch.x, batch.edge_index)
        
        # 只对训练集中的有标签节点计算损失
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 收集预测结果用于监控
        pred = out.argmax(dim=1)
        all_preds.extend(pred[batch.train_mask].cpu().numpy())
        all_labels.extend(batch.y[batch.train_mask].cpu().numpy())
    
    # 计算训练指标
    f1 = f1_score(all_labels, all_preds, average='binary')
    return total_loss / len(loader), f1

def validate(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        logits = out[data.val_mask]
        labels = data.y[data.val_mask]
        
        pred = logits.argmax(dim=1)
        
        # 计算多维度指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.cpu(), pred.cpu(), average='binary', zero_division=0
        )
        
        # 计算AUC（对于不平衡数据，PR-AUC比ROC-AUC更有信息量）
        if len(torch.unique(labels)) > 1:
            probs = F.softmax(logits, dim=1)[:, 1]
            roc_auc = roc_auc_score(labels.cpu(), probs.cpu())
        else:
            roc_auc = 0.5
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }