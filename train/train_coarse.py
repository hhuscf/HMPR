import argparse
import time
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.coarsenet import ImageCoarseNet, PointCloudCoarseNet
from eval.evaluate import recall_atN
from dataloader.dataset import PairwiseImageData, TestImageData, PairwisePCData, TestPCData
from loss.loss_fn import PairwiseMarginLoss

from utils.utils import (
    load_config,
    get_logger,
    count_parameters,
    load_checkpoint,
    get_current_lr,
    save_checkpoint,
)

# parameter parser
parser = argparse.ArgumentParser(description="PyTorch Training Coarse Net")
parser.add_argument("--work_path", type=str, required=True, help="working directory")
parser.add_argument("--config_path", type=str, required=True, help="config directory")
parser.add_argument("--lr", type=float, help="assign an new learning rate.")
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
parser.add_argument("--ckpt_file", type=str, help="checkpoint location")
args = parser.parse_args()

# load config from yaml file
config = load_config(args.config_path)

# logger, to record messages
work_path = args.work_path + '/' + config.arch_name
if not os.path.exists(work_path):
    os.makedirs(work_path)
logger = get_logger(log_file_name=work_path + "/log.txt", log_level="DEBUG", logger_name="CIFAR")

writer = SummaryWriter(comment='_' + config.arch_name + "_train.xxxx")

# variables for recording
best_prec = 0.


def test(test_loader, net, optimizer, epoch, device):
    global best_prec
    net.eval()

    features = []
    logger.info(" === Validation ===")

    with torch.no_grad():
        for data in test_loader:
            batch_feature = net(data.to(device))
            batch_feature = batch_feature.detach().cpu().numpy()
            features.append(batch_feature)
    features = np.vstack(features)

    # get some variables
    pos_items = test_loader.dataset.get_pos_items()
    num_of_each_run = test_loader.dataset.get_num_of_each_run()
    sum_num_of_each_run = [
        sum(num_of_each_run[:i]) for i in range(len(num_of_each_run))
    ]
    run_num = len(num_of_each_run)

    # compute evaluation metric
    recall_1s = []
    recall_1percents = []
    recall_25s = []

    pairs = ([i, j] for i in range(run_num) for j in range(i + 1, run_num))
    for i, j in pairs:
        st1 = sum_num_of_each_run[i]
        st2 = sum_num_of_each_run[j]
        end1 = st1 + num_of_each_run[i]
        end2 = st2 + num_of_each_run[j]

        feature_of_two_run = np.vstack((features[st1:end1], features[st2:end2]))
        pos_items_of_two_run = pos_items[(i, j)]

        recall_1, recall_1percent = recall_atN(
            feature_of_two_run, pos_items_of_two_run, N=1, Lp=config.loss.Lp
        )
        recall_25, _ = recall_atN(
            feature_of_two_run, pos_items_of_two_run, N=25, Lp=config.loss.Lp
        )
        recall_1s.append(recall_1)
        recall_1percents.append(recall_1percent)
        recall_25s.append(recall_25)

    # show and record test results
    recall_1 = np.mean(recall_1s)
    recall_1percent = np.mean(recall_1percents)
    recall_25 = np.mean(recall_25s)

    logger.info(f"   == test recall@1: {recall_1:.4f}")
    logger.info(f"   == test recall@1%: {recall_1percent:.4f}")
    logger.info(f"   == test recall@25: {recall_25:.4f}")

    writer.add_scalar("test_recall_1", recall_1, global_step=epoch)
    writer.add_scalar("test_recall_1%", recall_1percent, global_step=epoch)
    writer.add_scalar("test_recall_25", recall_25, global_step=epoch)

    # judge best testing
    is_best = recall_25 > best_prec
    if is_best:
        best_prec = recall_25

    # Save checkpoint.
    state = {
        "state_dict": net.state_dict(),
        "best_prec": best_prec,
        "last_epoch": epoch,
        "optimizer": optimizer.state_dict(),
    }

    save_checkpoint(state, is_best,  work_path + "/" + config.ckpt_name)
    logger.info(
        f"   == save checkpoint, recall@25={recall_25:.4}, is_best={is_best}, "
        f"best={best_prec:.4} =="
    )

    net.train()
    return recall_1


def train(loaders, net, criterion, optimizer, lr_scheduler, epoch, device):
    train_loader, test_loader = loaders
    start_time = time.time()
    net.train()

    train_loss_sum = 0
    batch_total_num = len(train_loader)

    logger.info(f" === start Epoch: [{epoch + 1}/{config.epochs}] ===")

    for batch_index, pairs in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train epoch ' + str(epoch+1), leave=False):
        loss = 0.0
        for pair_key, pair_data in pairs.items():
            y = 1 if pair_key == "pos_pair" else -1
            x = []
            for data in pair_data:
                x.append(net(data.to(device)))
            loss += criterion(*x, y)
        loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        if batch_index % config.show_freq == 0:
            logger.info(
                f"   == for step: [{batch_index + 1:5}/{batch_total_num:5}], "
                f"train loss: {loss.item():.3f} | "
                f"lr: {get_current_lr(optimizer):.5f}"
            )

        if (config.eval_freq != -1) and (batch_index + 1) % config.eval_freq == 0:
            recall_1 = test(test_loader, net, optimizer, epoch, device)
            lr_scheduler.step(recall_1)
            writer.add_scalar("learning_rate", get_current_lr(optimizer), global_step=epoch)

    # eval once at the end of epoch
    if config.eval_freq == -1:
        recall_1 = test(test_loader, net, optimizer, epoch+1, device)
        lr_scheduler.step(recall_1)
        writer.add_scalar("learning_rate", get_current_lr(optimizer), global_step=epoch+1)

    # record time for one epoch, train loss and train accuracy
    train_loss_avg = train_loss_sum / batch_total_num
    logger.info(f"   == cost time: {time.time() - start_time:.4f}s")
    logger.info(f"   == average train loss: {train_loss_avg:.3f}")
    writer.add_scalar("train_loss", train_loss_avg, global_step=epoch+1)

    return train_loss_avg


def main():
    global best_prec
    logger.info("\n\n" + "=" * 15 + " New Run " + "=" * 15)

    # define network
    if config.arch_name == 'image_coarse':
        net = ImageCoarseNet(config)
    elif config.arch_name == 'pointcloud_coarse':
        net = PointCloudCoarseNet(config)
    else:
        raise NotImplementedError("Not Implemented with other arch")
    logger.info(net)
    logger.info(f" == total parameters: {count_parameters(net)} ==")

    # CPU or GPU
    device = "cuda" if config.use_gpu else "cpu"
    logger.info(f" == will be trained on device: {device} ==")
    if device == "cuda":  # data parallel for multiple-GPU
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    net.to(device)

    # define loss and optimizer
    criterion = PairwiseMarginLoss(config.loss.a, config.loss.m, config.loss.Lp)

    optimizer = torch.optim.Adam(
        net.parameters(),
        config.optimize.base_lr,
        betas=config.optimize.betas,
        weight_decay=config.optimize.weight_decay,
        amsgrad=config.optimize.amsgrad,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_scheduler.factor,
        patience=config.lr_scheduler.patience,
        cooldown=config.lr_scheduler.cooldown,
        min_lr=config.optimize.base_lr / 10,
    )

    # resume from a checkpoint
    last_epoch = -1
    if args.resume:
        best_prec, last_epoch = load_checkpoint(args.ckpt_file, net, optimizer)
        lr_scheduler.step(best_prec)
    # overwrite learning rate
    if args.lr is not None:
        logger.info(f"learning rate is overwritten to {args.lr:.5}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr

    # load training and testing data loader
    if config.arch_name == 'image_coarse':
        train_datasets = PairwiseImageData(config=config)
        test_datasets = TestImageData(config=config)
    elif config.arch_name == 'pointcloud_coarse':
        train_datasets = PairwisePCData(config=config)
        test_datasets = TestPCData(config=config)
    else:
        raise NotImplementedError("Not Implemented with other arch")

    train_loader = DataLoader(
        train_datasets,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_datasets,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.workers,
    )

    # start training
    logger.info("==============  Start Training  ==============\n")
    for epoch in range(last_epoch + 1, config.epochs):
        random_int = torch.randint(200, 7000, (1,)).item()
        logger.info(f" === random number for dataloader this epoch: {random_int} ===")
        train_loader.dataset.shuffle_data(random_int)

        train(
            (train_loader, test_loader),
            net,
            criterion,
            optimizer,
            lr_scheduler,
            epoch,
            device,
        )

    # training finished
    logger.info(f"======== Training Finished.  best_test_acc: {best_prec:.3%} ========")
    writer.close()


if __name__ == "__main__":
    main()

