import argparse
import os
import pickle
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
from dataloader.dataset import PairwiseIMGPCData
from models.finenet import similarityFineNet
from utils.utils import (
    load_config,
    get_logger,
    count_parameters,
    load_checkpoint,
    get_current_lr,
    save_checkpoint,
)

# parameter parser
parser = argparse.ArgumentParser(description="PyTorch Training Fine Net")
parser.add_argument("--work_path", type=str, required=True, help="working directory")
parser.add_argument("--config_path", type=str, required=True, help="config directory")
parser.add_argument("--lr", type=float, help="assign an new learning rate.")
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
parser.add_argument("--ckpt_file", type=str, help="checkpoint location")
args = parser.parse_args()

# load config from yaml file
config = load_config(args.config_path)

# logger, to record messages
if not os.path.exists(args.work_path):
    os.makedirs(args.work_path)
logger = get_logger(log_file_name=args.work_path + "/log.txt", log_level="DEBUG", logger_name="CIFAR")

writer = SummaryWriter(comment="_fine_train.xxxx")

# variables for recording
best_recall1 = 0.


def read_pc(pc_path):
    pc_ndarray = np.fromfile(pc_path, dtype=np.float64).astype(np.float32)
    pc_ndarray.resize(pc_ndarray.shape[0] // 3, 3)
    pc_tensor = torch.from_numpy(pc_ndarray)
    return pc_tensor


def image_norm(image):
    image_normalizer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    return image_normalizer(image)


def read_img(img_path):
    # read data
    image_PIL = Image.open(img_path)
    box = (0, 0, 320, 200)
    image_PIL = image_PIL.crop(box)
    # prepare output
    image_tensor = image_norm(image_PIL)

    return image_tensor


def test(test_pickle_items, net, optimizer, epoch, device):
    global best_recall1
    logger.info(" === Validation ===\n")
    file_indices = test_pickle_items['file_index']
    pos_items = test_pickle_items['pos_index']
    coarse_index = test_pickle_items['coarse_index']
    run_num = len(file_indices)
    recall_1s = []
    recall_1percents = []
    pairs = [[i, j] for i in range(run_num) for j in range(i + 1, run_num)]

    for i, j in tqdm(pairs, total=len(pairs), desc='Testing'):
        pos_items_of_two_run = pos_items[(i, j)]
        coarse_index_of_two_run = coarse_index[(i, j)]
        indices_of_two_run = []
        indices_of_two_run.extend(file_indices[i])
        indices_of_two_run.extend(file_indices[j])
        assert len(pos_items_of_two_run) == len(coarse_index_of_two_run) == len(indices_of_two_run)
        total_num = len(indices_of_two_run)
        success_num = 0
        one_percent_num = max(int(round(len(indices_of_two_run) / 100.0)), 1)
        success_num1 = 0
        for idx in range(len(indices_of_two_run)):
            candidates = []
            targets = []
            candidates_img = []
            targets_img = []
            coarse_index_of_idx = coarse_index_of_two_run[idx].tolist()

            # use fine net refine
            # query pc
            folder, timestamp = indices_of_two_run[idx].split('/')
            pc_i_path = os.path.join(str(config.dataset.base_path), folder, str(config.dataset.pc_folder), timestamp + '.bin')
            pc_i_tensor = read_pc(pc_i_path).unsqueeze(0)
            # query img
            img_i_path = os.path.join(str(config.dataset.base_path), folder, str(config.dataset.image_folder), timestamp + '.png')
            img_i_tensor = read_img(img_i_path).unsqueeze(0)
            for jdx in range(len(coarse_index_of_idx)):
                folder, timestamp = indices_of_two_run[coarse_index_of_idx[jdx]].split('/')
                # candidate pc
                pc_j_path = os.path.join(str(config.dataset.base_path), folder, str(config.dataset.pc_folder), timestamp + '.bin')
                pc_j_tensor = read_pc(pc_j_path).unsqueeze(0)
                # candidate img
                img_j_path = os.path.join(str(config.dataset.base_path), folder, str(config.dataset.image_folder), timestamp + '.png')
                img_j_tensor = read_img(img_j_path).unsqueeze(0)

                candidates.append(pc_j_tensor)
                targets.append(pc_i_tensor)
                candidates_img.append(img_j_tensor)
                targets_img.append(img_i_tensor)
            candidates = torch.cat(candidates, dim=0)
            targets = torch.cat(targets, dim=0)
            candidates_img = torch.cat(candidates_img, dim=0)
            targets_img = torch.cat(targets_img, dim=0)

            net.eval()
            with torch.no_grad():
                out, _ = net(candidates.to(device), targets.to(device), candidates_img.to(device), targets_img.to(device))
                scores = out.detach().cpu().numpy()
            # re-rank the candidates
            coarse_index_of_idx = np.asarray(coarse_index_of_idx)[np.argsort(scores)[::-1]].tolist()

            if not pos_items_of_two_run[idx].any():
                total_num -= 1
                continue
            # calculate recall@1
            if set(coarse_index_of_idx[:1]) & set(pos_items_of_two_run[idx]):
                success_num += 1

            # calculate recall@1%
            if set(coarse_index_of_idx[:one_percent_num]) & set(pos_items_of_two_run[idx]):
                success_num1 += 1

        recall_1 = success_num / total_num
        recall_1percent = success_num1 / total_num
        logger.info(f" one combination recall@1: {recall_1:.4f}")

        recall_1s.append(recall_1)
        recall_1percents.append(recall_1percent)

    recall_1 = np.mean(recall_1s)
    recall_1percent = np.mean(recall_1percents)
    # record
    logger.info(f"   == final recall@1: {recall_1:.4f}")
    logger.info(f"   == final recall@1%: {recall_1percent:.4f}")
    writer.add_scalar("final_recall1", recall_1, global_step=epoch)
    writer.add_scalar("final_recall1percent", recall_1percent, global_step=epoch)

    # judge best testing
    is_best = recall_1 > best_recall1
    if is_best:
        best_recall1 = recall_1

    # Save checkpoint.
    state = {
        "state_dict": net.state_dict(),
        "best_prec": best_recall1,
        "last_epoch": epoch,
        "optimizer": optimizer.state_dict(),
    }

    save_checkpoint(state, is_best, args.work_path + "/" + config.ckpt_name)
    logger.info(
        f"   == save checkpoint, final recall@1={recall_1:.4}, is_best={is_best}, "
        f"best={best_recall1:.4} =="
    )

    net.train()
    return recall_1, recall_1percent


def train(loaders, test_pickle_items, net, optimizer, lr_scheduler, epoch, device):
    train_loader = loaders
    start_time = time.time()
    net.train()

    train_loss_sum = 0.
    batch_total_num = len(train_loader)

    logger.info(f" === start Epoch: [{epoch + 1}/{config.epochs}] ===")

    for batch_index, sample_batch in tqdm(enumerate(train_loader), total=len(train_loader),
                                          desc='Train epoch ' + str(epoch + 1), leave=False):
        net.train()
        optimizer.zero_grad()
        out, _ = net(sample_batch["pc0"], sample_batch["pc1"],sample_batch["img0"], sample_batch["img1"])
        labels = sample_batch["label"].float().to(device=device)
        loss = F.binary_cross_entropy(out, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

    # end one epoch
    test(test_pickle_items, net, optimizer, epoch + 1, device)
    lr_scheduler.step()
    writer.add_scalar("learning_rate", get_current_lr(optimizer), global_step=epoch + 1)

    # record time for one epoch, train loss and train accuracy
    train_loss_avg = train_loss_sum / batch_total_num
    logger.info(f"   == cost time: {time.time() - start_time:.4f}s")
    logger.info(f"   == average train loss: {train_loss_avg:.3f}")
    writer.add_scalar("train_loss", train_loss_avg, global_step=epoch + 1)

    return train_loss_avg


def main():
    global best_recall1
    logger.info("\n\n" + "=" * 15 + " New Run " + "=" * 15)

    # define network
    net = similarityFineNet(config)
    logger.info(net)
    logger.info(f" == total parameters: {count_parameters(net)} ==")

    # CPU or GPU
    device = "cuda" if config.use_gpu else "cpu"
    logger.info(f" == will be trained on device: {device} ==")
    if device == "cuda":  # data parallel for multiple-GPU
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    net.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        config.optimize.base_lr,
        betas=config.optimize.betas,
        weight_decay=config.optimize.weight_decay,
        amsgrad=config.optimize.amsgrad,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)

    # resume from a checkpoint
    last_epoch = -1
    if args.resume:
        _, last_epoch = load_checkpoint(args.ckpt_file, net, optimizer)
    # overwrite learning rate
    if args.lr is not None:
        logger.info(f"learning rate is overwritten to {args.lr:.5}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr

    # load training data loader
    train_loader = DataLoader(
        PairwiseIMGPCData(config=config),
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.workers,
        drop_last=True,
    )
    # testing data
    test_pickle_file = str(config.dataset.pickle_path) + '/' + str(config.dataset.test_pickle)
    with open(test_pickle_file, 'rb') as f:
        test_pickle_items = pickle.load(f)

    # start training
    logger.info("==============  Start Training  ==============\n")
    for epoch in range(last_epoch + 1, config.epochs):
        train(
            train_loader,
            test_pickle_items,
            net,
            optimizer,
            lr_scheduler,
            epoch,
            device,
        )

    # training finished
    logger.info(f"======== Training Finished.  best_final_recall@1: {best_recall1:.3%} ========")
    writer.close()


if __name__ == "__main__":
    main()
