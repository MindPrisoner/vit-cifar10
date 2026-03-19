import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import Config
#from datasets.mnist_dataset import get_mnist_dataloader
from datasets.cifar10_dataset import get_cifar10_dataloader
# from models.cnn import SimpleCNN
#from models.cnn import LeNet
#from models.alexnet import AlexNet
#from models.resnet import ResNet18  # 导入ResNet18
from models.vit import ViT
from utils.seed import set_seed
#假如TensorBoard
from utils.logger import get_writer
from engine.evaluator import evaluate
from engine.trainer import Trainer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
def main():

    config = Config()

    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_dataloader(
        config.batch_size,
        config.num_workers
    )
    #week1  simpleCNN
    # model = SimpleCNN().to(device)
    #week2  LeNet
    #model = LeNet().to(device)
    #model = AlexNet().to(device)
    #model = ResNet18().to(device)
    model = ViT(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr
    )
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=config.lr,
    #     momentum=0.9,
    #     weight_decay=5e-4  # L2正则化
    # )
    # 添加学习率调度器 - 每5轮降低为原来的0.1倍
    scheduler = StepLR(optimizer, step_size=20, gamma=0.2)

    #step 2
    writer = get_writer(config.log_dir)

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        writer,
        config,
        evaluate,
        scheduler
    )
    trainer.train()


if __name__ == "__main__":
    main()
