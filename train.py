import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.datamodule import DataModule
from src.models.classifier import ImageClassifier
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    with open("configs/train.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    dm = DataModule(
        cfg["paths"]["data_dir"],
        cfg["image_size"],
        cfg["batch_size"]
    )
    dm.setup()

    model = ImageClassifier(cfg["num_classes"]).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["scheduler"]["step_size"],
        gamma=cfg["scheduler"]["gamma"]
    )

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model, optimizer, criterion, DEVICE, cfg["mixed_precision"]
    )
    evaluator = Evaluator(model, DEVICE)

    for epoch in range(cfg["epochs"]):
        train_loss = trainer.train_epoch(dm.train_dataloader())
        val_acc = evaluator.evaluate(dm.val_dataloader())
        scheduler.step()

        print(f"Epoch {epoch+1} | Loss {train_loss:.4f} | Val Acc {val_acc:.4f}")

        save_checkpoint(
            model, optimizer, epoch,
            f"{cfg['paths']['checkpoint_dir']}/model_epoch_{epoch}.pth"
        )

if __name__ == "__main__":
    main()
