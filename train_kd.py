from email.policy import default
from re import S
from efficientvit.eval_seg_model import validation_epoch
from efficientvit.efficientvit.models.utils import resize
from efficientvit.efficientvit.seg_model_zoo import create_seg_model
from dataset.coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet
from dataset.voc import VOCDataTrainSet, VOCDataValSet
from dataset.camvid import CamvidTrainSet, CamvidValSet
from dataset.ade20k import ADETrainSet, ADEDataValSet
from dataset.cityscapes import CSTrainValSet
from utils.flops import cal_multi_adds, cal_param_size
from utils.score import SegmentationMetric
from utils.logger import setup_logger
from utils.distributed import *
from utils.sagan import Discriminator
from models.model_zoo import get_segmentation_model
from losses import *
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn as nn
import torch
import argparse
import time
import datetime
import os
import shutil
import sys
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


SAVE_PATH = "/home/aaryang/experiments/CIRKD/checkpoints/"

# EfficientViT models


def parse_args():
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation Training With Pytorch')
    parser.add_argument('--teacher-model', type=str, default='l1',
                        help='model name')
    parser.add_argument('--student-model', type=str, default='b0',
                        help='model name')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        help='dataset name')
    parser.add_argument('--teacher-weights-path', type=str, default='teacher_weights',
                        help='teacher weights')
    parser.add_argument('--student-weights-path', type=str, default =None,
                         help ='student weights')
    parser.add_argument('--data', type=str, default='./dataset/cityscapes/',
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[1024, 2048], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='ignore label')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')

    parser.add_argument("--kd-temperature", type=float,
                        default=1.0, help="logits KD temperature")
    parser.add_argument("--lambda-kd", type=float,
                        default=0., help="lambda_kd")

    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local-rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')

    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    
    parser.add_argument('--val-path',type =str, default='/home/c3-0/datasets/Cityscapes/leftImg8bit/val')
    parser.add_argument('--pretrained-student', type=bool, default=False)
    parser.add_argument('--lr-decay-iterations', type=int, default = 1)
    parser.add_argument('--task-lambda',type=float, default=0.25)
    parser.add_argument('--irregular-decay', type=bool, default = False)
    parser.add_argument('--lr-power', type = float, default = 0.9)
    parser.add_argument('--use-eff-val', type=bool, default = True)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(
            os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        if args.dataset == 'cityscapes':
            train_dataset = CSTrainValSet(args.data,
                                          list_path='./dataset/list/cityscapes/train.lst',
                                          max_iters=args.max_iterations*args.batch_size,
                                          crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(args.data,
                                        list_path='./dataset/list/cityscapes/val.lst',
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            train_dataset = VOCDataTrainSet(args.data, './dataset/list/voc/train_aug.txt', max_iters=args.max_iterations*args.batch_size,
                                            crop_size=(512,1024), scale=True, mirror=True)
            val_dataset = VOCDataValSet(
                args.data, './dataset/list/voc/val.txt')
        elif args.dataset == 'ade20k':
            train_dataset = ADETrainSet(args.data, max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=(512,512), scale=True, mirror=True)
            val_dataset = ADEDataValSet(args.data)
        elif args.dataset == 'camvid':
            train_dataset = CamvidTrainSet(args.data, './dataset/list/CamVid/camvid_train_list.txt', max_iters=args.max_iterations*args.batch_size,
                                           ignore_label=args.ignore_label, crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CamvidValSet(
                args.data, './dataset/list/CamVid/camvid_val_list.txt')
        elif args.dataset == 'coco_stuff_164k':
            train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt', max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                                  crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CocoStuff164kValSet(
                args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
        else:
            raise ValueError('dataset unfind')

        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(
            train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(
            train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(
            val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)


        # Do I need to modify this for my student model? --> Probably yes
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        # Pre-decided model, dataset and weights path --> modify to accept new  arguments later
        self.s_model = create_seg_model(args.student_model, args.dataset, pretrained = True, weight_url=args.student_weights_path)
        self.t_model = create_seg_model(args.teacher_model, args.dataset, pretrained=True, weight_url=args.teacher_weights_path)
       

        # All parameters of parent must be set with false
        for param in self.t_model.parameters():
            param.requires_grad = False

        # Set both to evaluation mode
        self.t_model.to("cuda")
        self.s_model.to("cuda")
        self.t_model.eval()
        self.s_model.eval()

        # create criterion
        x = torch.randn(1, 3, 512, 512).cuda()
        t_y = self.t_model(x)
        s_y = self.s_model(x)
        t_channels = t_y[-1].size(1)
        s_channels = s_y[-1].size(1)

        self.criterion = SegCrossEntropyLoss(
            ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = CriterionKD(
            temperature=args.kd_temperature).to(self.device)

        # List of all trainable params
        params_list = nn.ModuleList([])
        params_list.append(self.s_model)

        # Change to Adam?
        self.optimizer = torch.optim.SGD(params_list.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model,
                                                               device_ids=[
                                                                   args.local_rank],
                                                               output_device=args.local_rank)
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0

    def adjust_lr(self, base_lr, iter, max_iter, power):
        if self.args.irregular_decay :
            if iter < 30000 :
                cur_lr = base_lr*((1-float(iter//self.args.lr_decay_iterations)/30000)**(power))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr
            elif 30000 <= iter < 35000 :
                cur_lr = 0.000005*((1-float(iter-30000//self.args.lr_decay_iterations)/5000)**(power))
                cur_lr = max(cur_lr, 0.0000025)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr
            else :
                cur_lr = 0.0000025*((1-float(iter-35000//self.args.lr_decay_iterations)/5000)**(power))
                cur_lr = max(cur_lr, 0.000001)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr
        else :
            cur_lr = base_lr*((1-float(iter//self.args.lr_decay_iterations)/max_iter)**(self.args.lr_power))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr
        return cur_lr

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        rt = tensor.clone()
        if self.args.distributed :
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(
            args.max_iterations))

        self.s_model.train()

        for iteration, (images, targets, _) in enumerate(self.train_loader):
            if (not self.args.skip_val and iteration % val_per_iters == 0) or iteration == 0:
                if self.args.dataset == "cityscapes" :
                    val_mIoU = validation_epoch(self.args, self.s_model)
                    logger.info("{} mIoU = {}".format(iteration,val_mIoU))
                    self.s_model.train()
                else :
                    self.validation()
                    self.s_model.train()
                
            iteration = iteration + 1

            images = images.to(self.device)
            targets = targets.long().to(self.device)

            with torch.no_grad():
                t_outputs = self.t_model(images)

            s_outputs = self.s_model(images)

            if s_outputs.shape[-2:] != targets.shape[-2:]:
                s_outputs = resize(s_outputs, size=targets.shape[-2:])
            if t_outputs.shape[-2:] != targets.shape[-2:]:
                t_outputs = resize(t_outputs, size=targets.shape[-2:])

            task_loss = self.criterion(s_outputs, targets)
            kd_loss = torch.tensor(0.).cuda()


            # Change weight of lambda_kd = 1
            if self.args.lambda_kd != 0.:
                kd_loss = self.args.lambda_kd * \
                    self.criterion_kd(s_outputs, t_outputs)


            losses = kd_loss + args.task_lambda * task_loss
            lr = self.adjust_lr(base_lr=args.lr, iter=iteration-1,
                                max_iter=args.max_iterations, power=0.9)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()


            task_loss_reduced = self.reduce_mean_tensor(task_loss)
            kd_loss_reduced = self.reduce_mean_tensor(kd_loss)
           
            
            eta_seconds = ((time.time() - start_time) / iteration) * \
                (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || KD Loss: {:.4f}"
                    "|| Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'],
                        task_loss_reduced.item(),
                        kd_loss_reduced.item(),
                        str(datetime.timedelta(
                            seconds=int(time.time() - start_time))),
                        eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(SAVE_PATH, self.s_model, "b0", "cityscapes",
                                iteration, args.distributed, is_best=False)

        val_mIoU = validation_epoch(self.args, self.s_model)
        logger.info("{} mIoU = {}".format(self.args.max_iterations,val_mIoU))
        # Saving final model with max ites
        save_checkpoint(SAVE_PATH, self.s_model, "b0", "cityscapes",
                        args.max_iterations + 1, args.distributed, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / args.max_iterations))

    def validation(self):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(
            len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            
            if outputs.shape[-2:] != target.shape[-2:]:
                outputs = resize(outputs, size=target.shape[-2:])

            self.metric.update(outputs, target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc, mIoU))

        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(
                self.metric.total_correct).cuda().to(args.local_rank)
            sum_total_label = torch.tensor(
                self.metric.total_label).cuda().to(args.local_rank)
            sum_total_inter = torch.tensor(
                self.metric.total_inter).cuda().to(args.local_rank)
            sum_total_union = torch.tensor(
                self.metric.total_union).cuda().to(args.local_rank)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / \
                (2.220446049250313e-16 + sum_total_label)
            IoU = 1.0 * sum_total_inter / \
                (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                pixAcc.item() * 100, mIoU * 100))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
            save_checkpoint(SAVE_PATH, self.s_model, args.student_model, args.dataset,
                            args.max_iterations + 1, args.distributed, is_best=False)
        synchronize()


def save_npy(array, name):
    """Save Checkpoint"""
    if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
        directory = os.path.expanduser(args.save_dir)
        np.save(os.path.join(directory, name), array)


def save_checkpoint(save_dir, model, model_name, dataset, iteration, distributed, is_best=False):
    """Save Checkpoint"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = 'kd_{}_{}_{}.pth'.format(model_name, dataset, iteration)
    filename = os.path.join(save_dir, filename)

    if distributed:
        model = model.module

    torch.save(model.state_dict(), filename)

    if is_best:
        best_filename = 'kd_{}_{}_{}_best_model.pth'.format(
            model_name, dataset, iteration)
        best_filename = os.path.join(save_dir, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        synchronize()

    if args.student_weights_path :    
        logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='kd_{}_pretrained_{}_{}_batch_{}_lr_{}_decay_{}_task_lambda_{}_iters_{}_log.txt'.format(
            args.teacher_model, args.student_model, args.dataset,args.batch_size, args.lr, args.lr_decay_iterations, args.task_lambda, args.max_iterations))
    else :
        logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='kd_{}_{}_{}_batch_{}_lr_{}_decay_{}_task_lambda_{}_iters_{}_log.txt'.format(
            args.teacher_model, args.student_model, args.dataset,args.batch_size, args.lr, args.lr_decay_iterations, args.task_lambda, args.max_iterations))
        
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
