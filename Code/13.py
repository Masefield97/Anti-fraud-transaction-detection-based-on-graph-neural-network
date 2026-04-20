from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def detailed_evaluation(model, data, mask, name="Test"):
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        logits = out[mask]
        labels = data.y[mask]
        preds = logits.argmax(dim=1)
        
        # 混淆矩阵
        cm = confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])
        
        # 详细报告
        report = classification_report(
            labels.cpu(), preds.cpu(), 
            target_names=['Licit', 'Illicit'],
            output_dict=True
        )
        
        print(f"\n{'='*50}")
        print(f"Evaluation on {name} Set")
        print(f"{'='*50}")
        print(f"Confusion Matrix:")
        print(f"                Predicted")
        print(f"               Licit  Illicit")
        print(f"Actual Licit   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"       Illicit  {cm[1,0]:5d}  {cm[1,1]:5d}")
        print(f"\nClassification Report:")
        for label in ['licit', 'illicit']:
            print(f"  {label.capitalize()}: Precision={report[label]['precision']:.3f}, "
                  f"Recall={report[label]['recall']:.3f}, F1={report[label]['f1-score']:.3f}")
        print(f"\n  Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")
        
        return cm, report

# 分别在训练、验证、测试集上评估
train_cm, train_report = detailed_evaluation(model, data, data.train_mask, "Train")
val_cm, val_report = detailed_evaluation(model, data, data.val_mask, "Validation")
test_cm, test_report = detailed_evaluation(model, data, data.test_mask, "Test")