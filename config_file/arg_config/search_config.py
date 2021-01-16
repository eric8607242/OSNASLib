import argparse

def get_init_config():
    parser = argparse.ArgumentParser(description="Searching configuration")
    parser.add_argument("--title",                 type=str,    help="Experiment title")
    parser.add_argument("--resume",                type=str,    help="Resume path")
    parser.add_argument("--random-seed",           type=int,    help="Random seed")
    parser.add_argument("--device",                type=str,    default="cpu")
    
    # Supernet config
    parser.add_argument("--search-space",          type=str,    default="proxylessnas", help="Search spcae of different method")
    parser.add_argument("--sample-strategy",       type=str,    default="fair", help="Sampling strategy for training supernet")

    # Supernet training config
    parser.add_argument("-epochs",                 type=int,    default=120)

    parser.add_argument("--optimizer",             type=str,    default="sgd")
    parser.add_argument("--lr",                    type=float,  default=0.05)
    parser.add_argument("--weight_decay",          type=float,  default=0.0004)
    parser.add_argument("--momentum",              type=float,  default=0.9)

    parser.add_argument("--lr-scheduler",          type=str,    default="cosine")
    parser.add_argument("--decay-step",            type=int)
    parser.add_argument("--decay-ratio",           type=float)

    parser.add_argument("--alpha",           type=float)
    parser.add_argument("--beta",           type=float)

    # Datset config
    parser.add_argument("--dataset",               type=str,    default="cifar100", help="Name of dataset")
    parser.add_argument("--dataset-path",          type=str,    default="./data/")

    parser.add_argument("--classes",               type=int,    default=100,        help="Class number for classification")
    parser.add_argument("--input-size",            type=int,    default=32)

    parser.add_argument("--batch-size",            type=int,    default=128, help="Batch size")
    parser.add_argument("--num-workers",           type=int,    default=4)
    parser.add_argument("--train-portion",         type=float,  default=0.8)

    # Path config
    parser.add_argument("--logger-path",           type=str,    default="./logs/")
    parser.add_argument("--writer-path",           type=str,    default="./logs/")

    parser.add_argument("--lookup-table-path",     type=str,    default="./logs/lookup_table.json")

    parser.add_argument("--supernet-model-path",   type=str)
    parser.add_argument("--searched-model-path",   type=str)

    args = parser.parse_args()
    return args
    
