from datetime import datetime
import os
import argparse
import sys

import gain
import data

def set_available_gpus(gpus):
    if isinstance(gpus, list):
        gpu_str = ','.join(gpus)
    else:
        gpu_str = str(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

def train_handler(args):
    if args.gpus:
        set_available_gpus(args.gpus)

    output_dir = os.path.join(args.output_dir,
                                os.path.basename(args.dataset_path) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    heatmap_dir = os.path.join(output_dir, 'heatmaps')
    model_dir = os.path.join(output_dir, 'models')

    print('Creating Dataset...')
    rds = data.RawDataset(args.dataset_path)

    print('Creating Model...')
    model = gain.AttentionGAIN(rds, args.model, args.gradient_layer_name, learning_rate=args.learning_rate,
                          gpu=bool(args.gpus), heatmap_dir=heatmap_dir, saved_model_dir=model_dir, alpha=args.alpha)

    print('Starting Training')
    print('=================\n')
    model.train(args.num_epochs, pretrain_epochs=args.pretrain_epochs,
                pretrain_threshold=args.pretrain_threshold, test_every_n_epochs=args.test_every_n)
    print('\nTraining Complete')
    print('=================')

def infer_handler(args):
    raise NotImplementedError('Coming soon...')

def model_info_handler(args):
    raise NotImplementedError('Coming soon...')

def parse_args(argv):
    gpu_parent = argparse.ArgumentParser(add_help=False)
    gpu_parent.add_argument('--gpus', type=str, nargs='+',
        help='GPUs to run training on. Exclude for cpu training')

    data_parent = argparse.ArgumentParser(add_help=False)
    data_parent.add_argument('--dataset-path', type=str, required=True,
        help='The path to the dataset, formatted with data in different directories based on label')

    model_parent = argparse.ArgumentParser(add_help=False)
    model_parent.add_argument('--gradient-layer-name', type=str, default='features.34',
        help='The name of the layer to construct the heatmap from')
    model_parent.add_argument('--model', type=str, default='vgg19', choices=gain.available_models,
        help='The name of the underlying model to train')
    # TODO dynamically retrieve expected input size???
    model_parent.add_argument('--input-dims', type=int, nargs=2, default=(224,224),
        help='The dimensions to resize inputs to. Keep in mind that some models have a default input size.')

    parser = argparse.ArgumentParser(description='Implementation of GAIN using pytorch')

    subparser = parser.add_subparsers(help='The action to perform')

    train_parser = subparser.add_parser('train',  parents=[gpu_parent, data_parent, model_parent],
        help='Train a new model')
    train_parser.set_defaults(func=train_handler)
    train_parser.add_argument('--learning-rate', type=float, default=0.0005,
        help='Learning rate to plug into the optimizer')
    train_parser.add_argument('--test-every-n', type=int, default=5,
        help='Run a full iteration over the test epoch every n epochs')
    train_parser.add_argument('--alpha', type=float, default=1,
        help='The coefficied in Eq 6 that weights the attention mining loss in relation to the classification loss')
    train_parser.add_argument('--sigma', type=float, default=0.5,
        help='The threshold value used in Eq 6')
    train_parser.add_argument('--omega', type=float, default=10,
        help='The scaling value used in Eq 6')
    train_parser.add_argument('--pretrain-epochs', type=int, default=100,
        help='The number of epochs to train the network before factoring in the attention map')
    train_parser.add_argument('--pretrain-threshold', type=float, default=0.95,
        help='The accuracy value to pretrain to before factoring in the attention map loss')
    train_parser.add_argument('--num-epochs', type=int, default=50,
        help='The number of epochs to run training for')
    train_parser.add_argument('--batch-size', type=int, default=1,
        help='The batch size to use when training')
    train_parser.add_argument('--output-dir', type=str, default='./out',
        help='The output directory for training runs. A subdirectory with the modelname and timestamp is created')

    infer_parser = subparser.add_parser('infer', parents=[gpu_parent, model_parent],
        help='Run inference on a trained model')
    infer_parser.set_defaults(func=infer_handler)
    infer_parser.add_argument('--weights-file', type=str, required=True,
        help='The file containing the model\'s weights')
    infer_parser.add_argument('--image-path', type=str, required=True,
        help='The path to the image that you would like to classify')
    infer_parser.add_argument('--label', type=str,
        help='If this is set, a heatmap is only generated for this label. Otherwise, a heatmap is generated for all labels')
    infer_parser.add_argument('--output-dir', type=str,
        help='The directory to save heatmap outputs')
    infer_parser.add_argument('-c', '--concat-output', action='store_true',
        help='If this flag is specified, the output heatmaps are concatenated to one image')

    model_info_parser = subparser.add_parser('model', parents=[model_parent],
        help='Utility to print information about a model for easier layer selection')
    model_info_parser.set_defaults(func=model_info_handler)
    model_info_parser.add_argument('--weights-file', type=str,
        help='The optional choice to also print information about saved model weights')

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    args.func(args)
