
import os
import time
import torch
import numpy as np

from options.picamopt import Options
from loader.get_dataset import get_dataset
from models.create_model import CreateModel
from utils.helper import print_loss,tensorboard_visual
from tqdm import tqdm

from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True
@torch.no_grad()
def test(opt, model, loader, logger):
    TP, FP, FN, TN, total = 0,0,0,0,0
    label_tp, label_tn = 0,0

    for i, (img, template, label, path) in enumerate(loader):
        inputs = {'img': img, 'template': template}
        model.set_input(inputs, mode='test')
        output = model.inference().detach().cpu()
        # if label.data == 1:
        #     label_tp += 1
        # elif label.data == 0:
        #     label_tn += 1
        label_tp += (label == 1).cpu().sum()
        label_tn += (label == 0).cpu().sum()

        TP += ((output == 1) & (label == 1)).cpu().sum()
        TN += ((output == 0) & (label == 0)).cpu().sum()
        FP += ((output == 1) & (label == 0)).cpu().sum()
        FN += ((output == 0) & (label == 1)).cpu().sum()

        # if output.data == 1 and label.data == 1:
        #     TP += 1
        # elif output.data == 0 and label.data == 0:
        #     TN += 1
        # elif output.data == 1 and label.data == 0:
        #     FP += 1
        # elif output.data == 0 and label.data == 1:
        #     FN += 1
    Precision = float(TP) / (TP + FP) if (TP + FP) > 0 else 0
    Recall = float(TP) / (TP + FN) if (TP + FN) > 0 else 0

    Sensitive = float(TP) / (TP + FN) if (TP + FN) > 0 else 0
    Speicity = float(TN) / (TN + FP) if (TN + FP) > 0 else 0

    F1 = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) else 0

    acc = (float(TP) + float(TN)) / (label_tp + label_tn) * 100.0

    message = '\n------------------------results----------------------\n'
    message += '{:>10d}\t{:>10d}\n'.format(TP,label_tp)
    message += '{:>10d}\t{:>10d}\n'.format(TN,label_tn)
    message += '{:>10}\t{:>10.4f}\n'.format('acc:', acc)
    message += '{:>10}\t{:>10.4f}\n'.format('precision:', Precision)
    message += '{:>10}\t{:>10.4f}\n'.format('recall:', Recall)
    message += '{:>10}\t{:>10.4f}\n'.format('Specificity:', Speicity)
    message += '{:>10}\t{:>10.4f}\n'.format('Sensitivity:', Sensitive)
    message += '{:>10}\t{:>10.4f}\n'.format('F1-measure:', F1)
    message += '------------------------------------------------------\n'

    logger.write(message)
    return message, acc

def train(opt):
    device = torch.cuda.current_device()
    dataset = get_dataset(opt, train=1)
    test_dataset = get_dataset(opt, train=0)

    expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
    logger_name = os.path.join(expr_dir, 'logger.txt')

    logger = open(logger_name, 'wt')
    logger.write('experiment logger:\n')
    writer = SummaryWriter(os.path.join(expr_dir, 'train'))
    
    best_acc, best_epoch = 0,0

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=1,
        batch_size=opt.batch_size,
        shuffle=False
    )

    model = CreateModel(opt)

    if len(opt.gpu_ids) > 1:
        model = model.module

    if opt.load_model != '':
        model.load(opt.load_model)

    # msg, acc = test(opt, model, test_loader, logger)
    # print(msg)
    # exit(0)
    # writer.add_scalar('test/acc', acc, 0)
    total_epoch = opt.start_epochs + opt.epochs

    for epoch in range(opt.start_epochs, total_epoch):
        for i, (img, template, label, path) in enumerate(train_loader):
            inputs = {'img': img, 'template': template, 'label': label}
            model.set_input(inputs)
            model.optimize_parameters()
            loss = model.get_current_loss()
            msg = print_loss(loss, epoch, total_epoch, i, len(train_loader))

            if i % opt.display_freq == 0:
                msg = print_loss(loss, epoch, opt.start_epochs + opt.epochs, i, len(train_loader))
                tensorboard_visual(loss, epoch, opt.start_epochs + opt.epochs, i, len(train_loader), writer)
                print(msg)
                logger.write(msg)

        if epoch % opt.save_freq == 0:
            model.save(opt.name, epoch)

        if epoch % opt.val_freq == 0:
            msg, acc = test(opt, model, test_loader, logger)
            writer.add_scalar('test/acc', acc, epoch)
            if best_acc < acc:
                best_acc = acc
                best_epoch = epoch
                model.save(opt.name, 'best')
            print('test between ad and nc\n')
            print(msg)
            print('current best_acc is %.4f in epoch %d\n' % (best_acc, best_epoch))
            logger.write('current best_acc is %.4f in epoch %d\n' % (best_acc, best_epoch))

if __name__ == '__main__':
    trainOpt = Options().parse()
    #trainOpt.print_options()
    train(trainOpt)