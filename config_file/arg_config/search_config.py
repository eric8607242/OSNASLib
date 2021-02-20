import os
import argparse

def get_init_config():
    parser = argparse.ArgumentParser(description="Searching configuration")
    parser.add_argument("--title",                 type=str,                           help="Experiment title", required=True)
    parser.add_argument("--resume",                type=str,                           help="Resume path")
    parser.add_argument("--random-seed",           type=int,    default=42,            help="Random seed")
    parser.add_argument("--device",                type=str,    default="cpu")
    parser.add_argument("--ngpu",                  type=int,    default=4)
    
    # Search config
    parser.add_argument("--search-strategy",       type=str,    default="random_search", help="The way to search the best architecture[evolution, random_search, differentiable]")
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
    parser.add_argument("--sample-strategy",       type=str,    default="fair",        help="Sampling strategy for training supernet [fair, uniform, differentiable]")

    # Supernet training config
    parser.add_argument("-epochs",                 type=int,    default=120,           help="The epochs for supernet training")

    parser.add_argument("--optimizer",             type=str,    default="sgd",         help="Optimizer for supernet training")
    parser.add_argument("--lr",                    type=float,  default=0.05)
    parser.add_argument("--weight-decay",          type=float,  default=0.0004)
    parser.add_argument("--momentum",              type=float,  default=0.9)

    parser.add_argument("--lr-scheduler",          type=str,    default="cosine")
    parser.add_argument("--decay-step",            type=int)
    parser.add_argument("--decay-ratio",           type=float)

    parser.add_argument("--alpha",                 type=float)
    parser.add_argument("--beta",                  type=float)

    # Datset config
    parser.add_argument("--dataset",               type=str,    default="cifar100",    help="Name of dataset")
    parser.add_argument("--dataset-path",          type=str,    default="./data/",     help="Path to dataset")

    parser.add_argument("--classes",               type=int,    default=100,           help="Class number for classification")
    parser.add_argument("--input-size",            type=int,    default=32,            help="Input size of dataset")

    parser.add_argument("--batch-size",            type=int,    default=128,           help="Batch size")
    parser.add_argument("--num-workers",           type=int,    default=4)
    parser.add_argument("--train-portion",         type=float,  default=0.8)

    # Path config
    parser.add_argument("--root-path",             type=str,    default="./logs/")
    parser.add_argument("--logger-path",           type=str,    default="./logs/")
    parser.add_argument("--writer-path",           type=str,    default="./logs/tb/")

    parser.add_argument("--lookup-table-path",     type=str,    default="./lookup_table.json")

    parser.add_argument("--supernet-model-path",   type=str,    default="./best_supernet_model.pth")
    parser.add_argument("--searched-model-path",   type=str,    default="./searched_model_architecture.json")

    args = parser.parse_args()
    args = setting_path_config(args)

    return args

def setting_path_config(args):
    """
    Concatenate root path to each path argument
    """
    if not os.path.exists(os.path.join(args.root_path, args.title+"_{}".format(args.random_seed))):
        args.root_path = os.path.join(args.root_path, args.title+"_{}".format(args.random_seed))
        os.makedirs(args.root_path)

    args.lookup_table_path = os.path.join(args.root_path, args.lookup_table_path)
    args.supernet_model_path = os.path.join(args.root_path, args.supernet_model_path)
    args.searched_model_path = os.path.join(args.root_path, args.searched_model_path)
    return args

    
    
