import os 

from .search_config import get_search_config
from .evaluate_config import get_evaluate_config


def get_init_config(config_state):
    if config_state == "search":
        parser = get_search_config()
    elif config_state == "evaluate":
        parser = get_evaluate_config()

    parser = get_common_config(parser)
    args = setting_path_config(parser)

    return args


def get_common_config(parser):
    # Common config
    parser.add_argument("--title",                 type=str,                           help="Experiment title", required=True)
    parser.add_argument("--resume",                type=str,                           help="Resume path")
    parser.add_argument("--random-seed",           type=int,    default=42,            help="Random seed")
    parser.add_argument("--device",                type=str,    default="cuda")
    parser.add_argument("--ngpu",                  type=int,    default=1)

    # Dataset config
    parser.add_argument("--dataset",               type=str,    default="cifar100",    help="Name of dataset")
    parser.add_argument("--dataset-path",          type=str,    default="./data/",     help="Path to dataset")

    parser.add_argument("--classes",               type=int,    default=100,           help="Class number for classification")
    parser.add_argument("--input-size",            type=int,    default=32,            help="Input size of dataset")

    parser.add_argument("--batch-size",            type=int,    default=128,           help="Batch size")
    parser.add_argument("--num-workers",           type=int,    default=4)
    parser.add_argument("--train-portion",         type=float,  default=0.8)

    return parser


def setting_path_config(parser):
    """
    Concatenate root path to each path argument
    """
    # Path config
    parser.add_argument("--root-path",             type=str,    default="./logs/")
    parser.add_argument("--logger-path",           type=str,    default="./logs/")
    parser.add_argument("--writer-path",           type=str,    default="./logs/tb/")

    parser.add_argument("--lookup-table-path",     type=str,    default="./lookup_table.json")

    parser.add_argument("--supernet-model-path",   type=str,    default="./best_supernet_model.pth")
    parser.add_argument("--searched-model-path",   type=str,    default="./searched_model_architecture.npy")

    parser.add_argument("--hyperparameter-track-path", type=str,default="./hyperparameter_track.csv")

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.root_path, args.title+"_{}".format(args.random_seed))):
        args.root_path = os.path.join(args.root_path, args.title+"_{}".format(args.random_seed))
        os.makedirs(args.root_path)

    args.lookup_table_path = os.path.join(args.root_path, args.lookup_table_path)
    args.supernet_model_path = os.path.join(args.root_path, args.supernet_model_path)
    args.searched_model_path = os.path.join(args.root_path, args.searched_model_path)

    return args

