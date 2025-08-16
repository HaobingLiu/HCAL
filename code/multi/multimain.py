from utils import seed_everything
from multitrainer import ProtoTrainer
from dataloader.aircraft import train_loader_3, test_loader_3


if __name__ == "__main__":
    # 假设已实现的数据集（示例参数）

    seed_everything(9)
    trainer = ProtoTrainer(classes1=train_loader_3.dataset.len_1(),
                           classes2=train_loader_3.dataset.len_2(),
                           classes3=train_loader_3.dataset.len_3(),
                           )
    epoches = 1000
    # 训练循环
    for epoch in range(epoches):
        # 训练阶段
        train_loss, acc_1, acc_2, acc_3, train_hvr = trainer.train_epoch(train_loader_3, epoch)
        test_acc_1, test_acc_2, test_acc_3, test_hvr = trainer.test(test_loader_3, epoch + 1)
        print(
            f"\nEpoch {epoch + 1}/{epoches} | Train Loss: {train_loss:.4f} | Acc 1 : {acc_1 :.4f} |"
            f"Acc 2: {acc_2:.4f} | Acc 3: {acc_3:.4f} | HCVR: {train_hvr:.4f}")

        # 测试阶段（每5个epoch）
        # if (epoch + 1) % 5 == 0:


        print(
            f"\nTest Acc 1: {test_acc_1:.4f} | Test Acc 2: {test_acc_2:.4f} | Test Acc 3: {test_acc_3:.4f} | Test HVR: {test_hvr:.4f} \nBest"
            f" Acc 1: {trainer.acc1:.4f} | Best Acc 2: {trainer.acc2:.4f} | Best Acc 3: {trainer.acc3:.4f} | Best HVR: {trainer.hcvr_acc:.4f}")