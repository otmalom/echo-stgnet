import os
import os.path as osp
import argparse
import yaml
import json
from pprint import pprint
from lib.run.engine import get_engine

def parse_args():
    """
    parse args
    :return:args
    """
    # backbone_name: ResNet34
    # predictor_name: STCPredictor
    new_parser = argparse.ArgumentParser(
        description='PyTorch Cardiac Landmark Detector parser..')
    new_parser.add_argument('--device',     type=int,   default=0)
    new_parser.add_argument('--cfg_path',   type=str)
    new_parser.add_argument('--data_path',  type=str)
    new_parser.add_argument('--load_path',  type=str,   default=None)
    new_parser.add_argument('--proj_path',  type=str)
    new_parser.add_argument('--engine',     type=str)
    new_parser.add_argument('--loss_fn',    type=str)
    new_parser.add_argument('--backbone_name',     type=str)
    new_parser.add_argument('--predictor_name',    type=str)
    new_parser.add_argument('--num_kpts',   type=int,   default=32)
    new_parser.add_argument('--top_k',      type=int,   default=5)
    new_parser.add_argument('--sample_rate',type=float, default=1.0)
    new_parser.add_argument('--pretrained', action='store_true', default=False)
    new_parser.add_argument('--resume',     action='store_true', default=False)
    new_parser.add_argument('--visualize',  action='store_true', default=False)
    # 排他性参数
    group = new_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train',       action='store_true')
    group.add_argument('--evaluate',    action='store_true')
    group.add_argument('--eval_ckpts',  action='store_true')

    return new_parser.parse_args()


def main():
    # parse args and load config
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    # args=debug(args)  
    with open(args.cfg_path) as f:
        config = yaml.full_load(f)
    for k, v in vars(args).items():
        config[k] = v

    os.makedirs(args.proj_path, exist_ok=True)
    with open(osp.join(args.proj_path,'config.json'), "w") as f:
        f.write(json.dumps(config, ensure_ascii=False, indent=4, separators=(',', ':')))
    config['adj_dir'] = osp.join(args.data_path,'Points',f'{args.num_kpts}pts',f'adjmatrix_top{args.top_k}.txt')
    config['filelist'] = f'{config["data_path"]}/FileList.csv'
    if args.resume and osp.exists(osp.join(args.proj_path,'checkpoint.pth.tar')):
        config['resume']=True
        config['load_path']=osp.join(args.proj_path,'checkpoint.pth.tar')
    else:
        config['load_path']=None

    pprint(config)
    if args.train:
        agent = get_engine(config)
        agent.train()
    
    config['load_path']=osp.join(args.proj_path,'checkpoint_best.pth.tar')
    agent = get_engine(config)
    agent.evaluate()
    agent.test()

if __name__ == '__main__':
    main()
