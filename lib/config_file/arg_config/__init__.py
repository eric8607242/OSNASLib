import os
import csv
from datetime import datetime

from collections import OrderedDict

from .search_config import get_search_config
from .evaluate_config import get_evaluate_config


def get_init_config(config_state):
    if config_state == "search":
        parser = get_search_config()
    elif config_state == "evaluate":
        parser = get_evaluate_config()

    parser = get_common_config(parser)
    args = setting_path_config(parser, config_state)

    args.bn_track_running_stats = bool(args.bn_track_running_stats)

    hyperparameter_record(args)

    return args


def get_common_config(parser):
    # Common config
    parser.add_argument(
        "--title",
        type=str,
        help="Experiment title",
        required=True)
    parser.add_argument("--resume", type=str, help="Resume path")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ngpu", type=int, default=1)

    # Dataset config
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="Name of dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data/",
        help="Path to dataset")

    parser.add_argument("--classes", type=int, default=100,
                        help="Class number for classification")
    parser.add_argument(
        "--input-size",
        type=int,
        default=32,
        help="Input size of dataset")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-portion", type=float, default=0.8)

    return parser


def setting_path_config(parser, config_state):
    """
    Concatenate root path to each path argument
    """
    # Path config
    parser.add_argument("--root-path", type=str, default="./logs/")
    parser.add_argument("--logger-path", type=str, default="./logs/")
    parser.add_argument("--writer-path", type=str, default="./logs/tb/")

    parser.add_argument(
        "--checkpoint-path-root",
        type=str,
        default="./checkpoints/")
    parser.add_argument(
        "--lookup-table-path",
        type=str,
        default="./lookup_table.json")

    parser.add_argument(
        "--best-model-path",
        type=str,
        default="./best_model.pth")
    parser.add_argument(
        "--searched-model-path",
        type=str,
        default="./searched_model_architecture.npy")

    parser.add_argument(
        "--hyperparameter-tracker",
        type=str,
        default="./logs/{}_hyperparameter_tracker.csv".format(config_state))

    args = parser.parse_args()

    args.root_path = os.path.join(
        args.root_path,
        args.title +
        "_{}".format(
            args.random_seed))
    if not os.path.exists(args.root_path):
        os.makedirs(args.root_path)

    args.lookup_table_path = os.path.join(
        args.root_path, args.lookup_table_path)

    args.checkpoint_path_root = os.path.join(
        args.root_path, args.checkpoint_path_root, config_state)
    if not os.path.exists(args.checkpoint_path_root):
        os.makedirs(args.checkpoint_path_root)

    args.best_model_path = os.path.join(
        args.checkpoint_path_root, args.best_model_path)
    args.searched_model_path = os.path.join(
        args.root_path, args.searched_model_path)

    return args


def hyperparameter_record(args):
    current_time = datetime.now()
    current_time_str = current_time.strftime("%d/%m/%Y %H:%M:%S")

    hyperparameter_dict = OrderedDict(time=current_time_str)
    hyperparameter_dict.update([(k, v) for k, v in vars(args).items()])

    record_header = False
    if not os.path.exists(args.hyperparameter_tracker):
        # If file do not exist, we should record the header first
        record_header = True

    with open(args.hyperparameter_tracker, "a", newline="") as csvfile:
        dw = csv.DictWriter(csvfile, fieldnames=hyperparameter_dict.keys())
        if record_header:
            dw.writeheader()

        dw.writerow(hyperparameter_dict)
