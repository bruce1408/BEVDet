# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmcv import Config, DictAction

from mmdet3d.models import build_model

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default="/home/cuidongdong/BEVDet/configs/bevdepth/bevdepth4d-r50.py",
                        help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        # default=[40000, 4],
        default=[256, 704],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='point',
        choices=['point', 'image', 'multi'],
        help='input data modality')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def construct_input(input_shape, depth=True):
    rot = torch.eye(3).float().cuda().view(1, 3, 3)
    rot = torch.cat([rot for _ in range(6)], axis=0).view(1, 6, 3, 3)

    # input = dict(img_inputs=[
    #     torch.ones(()).new_empty((1, 6, 3, *input_shape)).cuda(),
    #     rot,
    #     torch.ones((1, 6, 3)).cuda(),
    #     rot,
    #     rot,
    #     torch.ones((1, 6, 3)).cuda()
    #     # torch.ones((1, 6, 128, 128)).cuda()
    # ])

    # rot = torch.eye(3).float().cuda().view(1, 3, 3)
    # rot = torch.cat([rot for _ in range(6)], axis=0).view(1, 6, 3, 3)
    input = dict(img_inputs=[
        torch.ones(()).new_empty(
            (1, 6, 3, *input_shape)).cuda(),
        rot,
        torch.ones((1, 6, 3)).cuda(),
        rot,
        rot,
        torch.ones((1, 6, 3)).cuda(), None])

    return input


def main():
    args = parse_args()

    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)
    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3,) + tuple(args.shape)
        else:
            raise ValueError('invalid input shape')
    elif args.modality == 'multi':
        raise NotImplementedError(
            'FLOPs counter is currently not supported for models with '
            'multi-modality input')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print("model eval is over")

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                model.__class__.__name__))

    print("before get flops")
    flops, params = get_model_complexity_info(model, input_shape, input_constructor=construct_input)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    print()


if __name__ == '__main__':
    main()
    # result = construct_input([256, 704])







