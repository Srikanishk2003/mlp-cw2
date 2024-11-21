import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    # Arguments
    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Epoch to continue training from')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018, help='Random seed')
    parser.add_argument('--image_num_channels', nargs="?", type=int, default=3, help='Number of channels in image')
    parser.add_argument('--image_height', nargs="?", type=int, default=32, help='Height of image')
    parser.add_argument('--image_width', nargs="?", type=int, default=32, help='Width of image')
    parser.add_argument('--num_stages', nargs="?", type=int, default=3, help='Number of convolutional stages')
    parser.add_argument('--num_blocks_per_stage', nargs="?", type=int, default=5, help='Blocks per stage')
    parser.add_argument('--num_filters', nargs="?", type=int, default=16, help='Filters per convolutional layer')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='Total epochs')
    parser.add_argument('--num_classes', nargs="?", type=int, default=100, help='Classes')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1", help='Experiment name')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=True, help='Flag to use GPU')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=0, help='Weight decay for Adam')
    parser.add_argument('--block_type', type=str, default='conv_block', help='Type of convolutional blocks')
    
    # ** New Argument for Learning Rate
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the experiment')

    # Parsing arguments
    args = parser.parse_args()
    print(args)
    return args

