checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,  # 每隔50batch打印结果
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# find_unused_parameters = True  # 是否查找模型中未使用的参数,在mobileNet-v2的时候使用，在调整最后一层的输出的时候也不太需要这个了
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
