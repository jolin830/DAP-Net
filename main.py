
import argparse
import os, csv
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torch.nn.functional as F
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
# from ours_sk import BDSubject_SK
from ours_sk_normal import BDSubject_SK
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
import random


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', default="", type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')  
    parser.add_argument('--model', default='pointMLP_DAP', help='model name')
    parser.add_argument('--num_classes', default=25, type=int, help='default value for classes') 
    parser.add_argument('--epoch', default=150, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')    
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')    
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--seed', type=int, default=1,help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    parser.add_argument('--data-path', default='UniMM-HAR/npz/CSub', type=str, help='dataset')
    parser.add_argument('--inputC', default=3, help='input data channel')
    parser.add_argument('--fast_scale', default=5, help='')
    parser.add_argument('--quantile', default=0.3, help='')
    parser.add_argument('--save-dir', default="checkpoints/", type=str, help='save')   # Cset
    parser.add_argument('--note', default="point change", type=str, help='save')   


    return parser.parse_args()



def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is not None:
        # torch.manual_seed(args.seed)
        init_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.save_path = args.save_dir + args.model + message
    if not os.path.isdir(args.save_path):
        mkdir_p(args.save_path)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.save_path, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    net = models.__dict__[args.model](num_classes=args.num_classes, inputC = args.inputC, fast_scale = args.fast_scale, quantile = args.quantile)
    criterion = cal_loss
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None
    save_args(args)

    if not os.path.isfile(os.path.join(args.save_path, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.save_path, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
        loss_logger = Logger(os.path.join(args.save_path, 'loss.txt'), title="Losses" + args.model)
        loss_logger.set_names(["Epoch", "Train_cls", "Train_offset", "Train_moment", "Val_cls", "Val_offset", "Val_moment"])

    if args.checkpoint:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])

        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')  
    dataset = BDSubject_SK(
        root=args.data_path,
        inputC = args.inputC,
        train=True
    )

    dataset_test = BDSubject_SK(
        root=args.data_path,
        inputC = args.inputC,
        train=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=init_seed)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, worker_init_fn=init_seed)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100, last_epoch=start_epoch - 1)


    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))

        train_out = train(net, train_loader, optimizer, criterion, device)  
        test_out = validate(net, test_loader, criterion, device)

        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            net, epoch, path=args.save_path, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                    train_out["loss"], train_out["acc_avg"], train_out["acc"],
                    test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")

        printf(
            f"[Train] total={train_out['loss']} "
            f"cls={train_out['loss_cls']} "
            f"acc={train_out['acc']}%"
        )

        printf(
            f"[Val  ] total={test_out['loss']} "
            f"cls={test_out['loss_cls']} "
            f"acc={test_out['acc']}%"
        )
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)


def train(net, trainloader, optimizer, criterion, device):
    net.train()
    cls_loss_sum = 0.0
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label, _) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        optimizer.zero_grad()
        logits = net(data)
        loss_cls = criterion(logits, label)
        loss = loss_cls 
        loss.backward()
        optimizer.step()

        cls_loss_sum += loss_cls.item()

        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "loss_cls": round(cls_loss_sum / (batch_idx + 1), 4),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = cls_loss_sum = offset_loss_sum = moment_loss_sum = 0.0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label,_) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            logits = net(data)
            loss_cls = criterion(logits, label)
            loss = loss_cls

            test_loss += loss.item()
            cls_loss_sum += loss_cls.item()

            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "loss_cls": round(cls_loss_sum / (batch_idx + 1), 4),
        "loss_offset": round(offset_loss_sum / (batch_idx + 1), 4),
        "loss_moment": round(moment_loss_sum / (batch_idx + 1), 4),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }



if __name__ == '__main__':
    main()
