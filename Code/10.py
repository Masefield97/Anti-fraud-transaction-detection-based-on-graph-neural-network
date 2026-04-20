def train_model(model, train_loader, data, epochs=100, lr=0.001, patience=20):
    device = next(model.parameters()).device
    data = data.to(device)
    
    # 加权交叉熵损失
    class_weights = compute_class_weight('balanced', classes=[0, 1], 
                                         y=labels[labels != -1])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    
    # Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, data, device)
        
        # 学习率调度
        scheduler.step(val_metrics['f1'])
        
        # 早停与模型保存
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch}: New best model! Val F1={val_metrics['f1']:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
                  f"Val P: {val_metrics['precision']:.3f} R: {val_metrics['recall']:.3f} F1: {val_metrics['f1']:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model