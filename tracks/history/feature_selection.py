"""Convenience entry point for the integrated seed-0 top-1 pipeline."""
from train import main

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--data_dir',required=True); p.add_argument('--k',type=int,default=16); p.add_argument('--lr',type=float,default=.001); p.add_argument('--l2',type=float,default=1e-6); p.add_argument('--batch_size',type=int,default=8192); p.add_argument('--selection_epochs',type=int,default=7); p.add_argument('--selection_weight',type=float,default=.25); p.add_argument('--epochs',type=int,default=15); main(p.parse_args())
