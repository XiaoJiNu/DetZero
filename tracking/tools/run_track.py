import time
import torch
import argparse
import shutil
import sys
import uuid

import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for local_path in (REPO_ROOT, REPO_ROOT / 'utils', REPO_ROOT / 'tracking'):
    sys.path.insert(0, str(local_path))

from detzero_utils.config_utils import cfg, cfg_from_yaml_file, log_cfg_info
from detzero_utils.common_utils import create_logger, get_log_info

from detzero_track.models import build_model, run_model
from detzero_track.datasets import build_dataloader
from tools.external_detector.pipeline import rename_noreplace


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="tools/cfgs/tk_model_cfgs/waymo_detzero_track.yaml", help='load the config for tracking')
    parser.add_argument('--data_path', type=str, required=True, help='data path of detection results')
    parser.add_argument('--output_path', type=Path, required=True, help='fresh output directory')
    parser.add_argument('--root_path', type=Path, help='Waymo data root')
    parser.add_argument('--workers', type=int, default=4, help='num of cpu for running track module')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('--split', type=str, default='val', help='different data split')
    parser.add_argument('--save_log', action="store_true", help="save log info into .txt")
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    return args, cfg

def main():
    args, cfg = parse_config()
    if args.output_path.exists() or args.output_path.is_symlink():
        raise FileExistsError(args.output_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = args.output_path.with_name(
        '.%s.tmp-%s' % (args.output_path.name, uuid.uuid4().hex)
    )
    stage.mkdir()
    log_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    if args.save_log:
        log_file = cfg.ROOT_DIR/'log'
        if not log_file.is_dir():
            log_file.mkdir(parents=True, exist_ok=True)
        log_file = log_file/(__file__.split('/')[-1]+'-'+log_time+'.txt')
    else:
        log_file = None

    logger = create_logger(log_file)
    logger.info(get_log_info('DetZero Tracking Module'))

    cfg_str_list = list()
    for key, val in vars(args).items():
        cfg[key.upper()] = val

    log_cfg_info(cfg, cfg_str_list, logger)

    try:
        dataset, dataloader = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            data_path=args.data_path,
            log_time=log_time,
            batch_size=args.batch_size,
            workers=args.workers,
            split=args.split,
            logger=logger,
            root_path=args.root_path,
            output_path=stage,
        )

        model = build_model(cfg.MODEL, logger)
        run_model(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            workers=args.workers,
            cfgs=cfg,
            logger=logger,
        )
        rename_noreplace(stage, args.output_path)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    logger.info(get_log_info('DetZero Tracking module Finished!'))

if __name__ == '__main__':
    main()
