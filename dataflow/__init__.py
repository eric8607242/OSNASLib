import sys

def get_dataloader(dataset_name, dataset_path, input_size, batch_size, num_workers, train_portion=1):
    dataloader_builder = getattr(sys.modules[__name__], f"get_{dataset_name}_dataloader")

    return dataloader_builder(dataset_path, input_size, batch_size, num_workers, train_portion)


# Import customizing module
from .cifar import get_cifar100_dataloader, get_cifar10_dataloader
from .imagenet import get_imagenet_dataloader
from .face_triplet import get_face_triplet_dataloader
from .face_angular import get_face_angular_dataloader
