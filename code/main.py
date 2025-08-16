from utils import seed_everything
from trainer import ProtoTrainer
from dataloader.cifar100 import train_loader,test_loader


if __name__ == "__main__":
    seed_everything(9)
    trainer = ProtoTrainer(num_classes=train_loader.dataset.len_fine(),
                           coarse_classes=train_loader.dataset.len_coarse())
    epoches = 500
    for epoch in range(epoches):
        train_loss, fine_acc, coarse_acc, train_hvr = trainer.train_epoch(train_loader, epoch)
        test_fine_acc, test_coarse_acc, test_hvr = trainer.test(test_loader, epoch + 1)
        print(
            f"\nEpoch {epoch + 1}/{epoches} | Train Loss: {train_loss:.4f} | Fine Acc: {fine_acc :.4f} | Coarse "
            f"Acc: {coarse_acc:.4f} | HCVR: {train_hvr:.4f}")


        print(
            f"\nTest Fine Acc: {test_fine_acc:.4f} | Test Coarse Acc: {test_coarse_acc:.4f} | Test HVR: {test_hvr:.4f} \nBest Fine"
            f" Acc: {trainer.fine_acc:.4f} | Best Coarse Acc: {trainer.coarse_acc:.4f} | Best HVR: {trainer.hcvr_acc:.4f}")