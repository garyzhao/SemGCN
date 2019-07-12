from __future__ import print_function, absolute_import, division

import time
import argparse
import numpy as np
import os.path as path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.utils import AverageMeter
from common.data_utils import read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from common.camera import camera_to_world, image_coordinates
from common.visualization import render_animation


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', required=True,
                        help='checkpoint to evaluate (file name)')

    # Model arguments
    parser.add_argument('-a', '--architecture', default='gcn', type=str, metavar='NAME',
                        help='architecture of the model (gcn or linear)')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('--hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('--non_local', dest='non_local', action='store_true', help='if use non-local layers')
    parser.set_defaults(non_local=False)
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')

    # Visualization
    parser.add_argument('--viz_subject', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz_action', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz_camera', type=int, default=0, metavar='N', help='camera to render')
    parser.add_argument('--viz_video', type=str, default=None, metavar='PATH', help='path to input video')
    parser.add_argument('--viz_skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz_output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz_bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz_limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz_downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz_size', type=int, default=5, metavar='N', help='image size')

    args = parser.parse_args()
    return args


def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    cudnn.benchmark = True
    device = torch.device("cuda")

    # Create model
    print("==> Creating model...")

    if args.architecture == 'linear':
        from models.linear_model import LinearModel, init_weights
        num_joints = dataset.skeleton().num_joints()
        model_pos = LinearModel(num_joints * 2, (num_joints - 1) * 3).to(device)
        model_pos.apply(init_weights)
    elif args.architecture == 'gcn':
        from models.sem_gcn import SemGCN
        from common.graph_utils import adj_mx_from_skeleton
        p_dropout = (None if args.dropout == 0.0 else args.dropout)
        adj = adj_mx_from_skeleton(dataset.skeleton())
        model_pos = SemGCN(adj, args.hid_dim, num_layers=args.num_layers, p_dropout=p_dropout,
                           nodes_group=dataset.skeleton().joints_group() if args.non_local else None).to(device)
    else:
        raise KeyError('Invalid model architecture')

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    # Resume from a checkpoint
    ckpt_path = args.evaluate

    if path.isfile(ckpt_path):
        print("==> Loading checkpoint '{}'".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch']
        error_best = ckpt['error']
        model_pos.load_state_dict(ckpt['state_dict'])
        print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
    else:
        raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))

    print('==> Rendering...')

    poses_2d = keypoints[args.viz_subject][args.viz_action]
    out_poses_2d = poses_2d[args.viz_camera]
    out_actions = [args.viz_camera] * out_poses_2d.shape[0]

    poses_3d = dataset[args.viz_subject][args.viz_action]['positions_3d']
    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
    out_poses_3d = poses_3d[args.viz_camera]

    ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()

    input_keypoints = out_poses_2d.copy()
    render_loader = DataLoader(PoseGenerator([out_poses_3d], [out_poses_2d], [out_actions]), batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)

    prediction = evaluate(render_loader, model_pos, device, args.architecture)[0]

    # Invert camera transformation
    cam = dataset.cameras()[args.viz_subject][args.viz_camera]
    prediction = camera_to_world(prediction, R=cam['orientation'], t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=0)
    ground_truth[:, :, 2] -= np.min(ground_truth[:, :, 2])

    anim_output = {'Regression': prediction, 'Ground truth': ground_truth}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
    render_animation(input_keypoints, anim_output, dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'],
                     args.viz_output, limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                     input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                     input_video_skip=args.viz_skip)


def evaluate(data_loader, model_pos, device, architecture):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    predictions = []

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        if architecture == 'linear':
            outputs_3d = model_pos(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
            outputs_3d = torch.cat([torch.zeros(num_poses, 1, outputs_3d.size(2)), outputs_3d], 1)  # Pad hip joint
        else:
            outputs_3d = model_pos(inputs_2d).cpu()
            outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-centre the root (hip)

        predictions.append(outputs_3d.numpy())

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.val, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()

    return np.concatenate(predictions), epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


if __name__ == '__main__':
    main(parse_args())
