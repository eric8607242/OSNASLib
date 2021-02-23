import os
import argparse

def get_evaluate_config():
    parser = argparse.ArgumentParser(description="Evaluating configuration")
    # Search config
    parser.add_argument("--search-strategy",       type=str,    default="random_search", help="The way to search the best architecture[evolution, random_search, differentiable]")
    parser.add_argument("--target-hc",             type=int,    default=100,           help="Target hardware constraint")
    parser.add_argument("--hc-weight",             type=float,  default=0.005,         help="The weight of hardware constraint objective")

    parser.add_argument("--info-metric",           type=str,    default="flops",       help="HC objective for searching")

    # Supernet config
    parser.add_argument("--search-space",          type=str,    default="proxylessnas", help="Search spcae in different paper [proxylessnas, fbnet_s, fbnet_l, spos]")

    # Evaluate training config
    parser.add_argument("--epochs",                 type=int,    default=120,           help="The epochs for supernet training")

    parser.add_argument("--optimizer",             type=str,    default="sgd",         help="Optimizer for supernet training")
    parser.add_argument("--lr",                    type=float,  default=0.05)
    parser.add_argument("--weight-decay",          type=float,  default=0.0004)
    parser.add_argument("--momentum",              type=float,  default=0.9)

    parser.add_argument("--lr-scheduler",          type=str,    default="cosine")
    parser.add_argument("--decay-step",            type=int)
    parser.add_argument("--decay-ratio",           type=float)

    parser.add_argument("--alpha",                 type=float)
    parser.add_argument("--beta",                  type=float)

    return parser

    
    
