import torch
from src.modules.avocode_module import AvocodoModule
from src.utils.conf import get_hydra_cnf
cfg = get_hydra_cnf(config_dir="./src/conf", config_name="config")

from avocodo import (
    AvocodoConfig,
    Avocodo
)

def avocodo_plmodule_to_pytorch(checkpoint_path, output_path):
    model = AvocodoModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()
    torch.save(model.model.state_dict(), output_path)
    
    avo_cfg = AvocodoConfig()
    model = Avocodo(avo_cfg)
    model.load_state_dict(torch.load(output_path))
    
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/00002/checkpoint-epoch=0426-val_l1_loss=0.0823.ckpt")
    parser.add_argument("--output", type=str, default="./data/avocodo_v1.pth")
    
    args = parser.parse_args()
    
    avocodo_plmodule_to_pytorch(args.checkpoint, args.output)