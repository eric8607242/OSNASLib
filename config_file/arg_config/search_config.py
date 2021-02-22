import os
import argparse

def get_search_config():
    parser = argparse.ArgumentParser(description="Searching configuration")
    # Search config
    parser.add_argument("--search-strategy",       type=str,    default="random_search", help="The way to search the best architecture[evolution, random_search, differentiable, differentiable_gumbel]")

    parser.add_argument("--target-hc",             type=int,    default=100,           help="Target hardware constraint")
    parser.add_argument("--hc-weight",             type=float,  default=0.005,         help="The weight of hardware constraint objective")

    parser.add_argument("--info-metric",           type=str,    default="flops",       help="HC objective for searching")

    # Random search
    parser.add_argument("--random-iteration",      type=int,    default=1000,          help="The network architectures sample num for random search")
    # Evolution algorithm
    parser.add_argument("--generation_num",        type=int,    default=20,            help="Generation num for evolution algorithm")
    parser.add_argument("--population",            type=int,    default=60,            help="Population size for evoluation algorithm")
    parser.add_argument("--parent-num",            type=int,    default=10,            help="Parent size for evolution algorithm")
    # Differentiable 
    parser.add_argument("--a-optimizer",             type=str,    default="sgd",         help="Optimizer for supernet training")
    parser.add_argument("--a-lr",                    type=float,  default=0.05)
    parser.add_argument("--a-weight-decay",          type=float,  default=0.0004)
    parser.add_argument("--a-momentum",              type=float,  default=0.9)

    parser.add_argument("--a-decay-step",            type=int)
    parser.add_argument("--a-decay-ratio",           type=float)

    parser.add_argument("--a-alpha",                 type=float)
    parser.add_argument("--a-beta",                  type=float)




    # Supernet config
    parser.add_argument("--search-space",          type=str,    default="proxylessnas", help="Search spcae in different paper [proxylessnas, fbnet_s, fbnet_l, spos]")
    parser.add_argument("--sample-strategy",       type=str,    default="uniform",        help="Sampling strategy for training supernet [fair, uniform, differentiable, differentiable_gumbel]")

    # Supernet training config
    parser.add_argument("-epochs",                 type=int,    default=120,           help="The epochs for supernet training")

    parser.add_argument("--optimizer",             type=str,    default="sgd",         help="Optimizer for supernet training")
    parser.add_argument("--lr",                    type=float,  default=0.0225)
    parser.add_argument("--weight-decay",          type=float,  default=0.00004)
    parser.add_argument("--momentum",              type=float,  default=0.9)

    parser.add_argument("--lr-scheduler",          type=str,    default="cosine")
    parser.add_argument("--decay-step",            type=int)
    parser.add_argument("--decay-ratio",           type=float)

    parser.add_argument("--alpha",                 type=float)
    parser.add_argument("--beta",                  type=float)

    return parser

    
    
