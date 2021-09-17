import yaml
from pprint import pprint

configs = {}

configs['project'] = {
    'proj_dir': 'project/',
    'model_dir': 'models/',
    'cfg_fname': 'configs.yml',
    'train_log': {
        'path': 'train.log',
        'screen_intvl': 20,
        'headers': {
            'Loss': ':3.4f',
            'bkb_grad': ':3.2f',
            'head_grad': ':3.2f',
        },
    },
    'val_log': {
        'path': 'val.log',
        'screen_intvl': 1,
        'headers': {
            'Loss': ':3.4f',
        },
    },
    'eval_log': {
        'path': 'eval.log',
        'screen_intvl': 1,
        'headers': {
            'Loss': ':3.4f',
        },
    },
    'val_intvl': 40,
    'eval_intvl': 40,
    'save_iters': [500, 1000, 1500, 2000],
}

configs['data'] = {
    'base': {
        'dataset': {
            'type': 'PenstateDataset',
            'ann_file': 'data/data_16k.lst',
            'seed': 10086,
            'split': [7, 2, 1],
            'duration': [16, 18],
            'sample_rate': 16000,
        },
        'dataloader': {
            'type': 'DataLoader',
            'batch_size': 1,
            'num_workers': 1,
            'pin_memory': False,
            'shuffle': False,
            'drop_last': False,
        },
    },
    'train': {
        'dataset': {
            'mode': 'train',
        },
        'dataloader': {
            'batch_size': 128,
            'num_workers': 4,
            'shuffle': True,
            'drop_last': True,
        },
    },
    'val': {
        'dataset': {
            'mode': 'val',
        },
        'dataloader': {},
    },
    'eval': {
        'dataset': {
            'mode': 'eval',
        },
        'dataloader': {},
    },
}

configs['model'] = {
    'base': {
        'net': {},
        'optimizer': {
            'type': 'SGD',
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        },
        'scheduler': {
            'type': 'MultiStepLR',
            'milestones': [4000, 6000, 8000],
            'gamma': 0.1,
        },
    },
    'backbone': {
        'net': {
            'type': 'VoiceFeatNet',
            'sample_rate': 16000,
            'n_fft': 512,
            'n_mels': 64,
            'cnn_channel': 64,
            'feat_dim': 64,
        },
    },
    'head': {
        'net': {
            'type': 'AvgPool_Linear',
            'output_dim': 20370,
        },
    },
}


with open('base_configs.yml', 'w') as f:
    yaml.dump(configs, f, sort_keys=False, default_flow_style=None, width=50)

pprint(configs)
