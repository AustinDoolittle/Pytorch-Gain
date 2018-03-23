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

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True,
        help='The path to the dataset, formatted with data in different directories based on label')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
        help='Learning rate to plug into the optimizer')
    parser.add_argument('--gpus', type=str, nargs='+',
        help='GPUs to run training on. Exclude for cpu training')
    parser.add_argument('--gradient-layer-name', type=str, default='features.34',
        help='The name of the layer to construct the heatmap from')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='The number of epochs to run training for')
    parser.add_argument('--output-dir', type=str, default='./out',
        help='The output directory for training runs. A subdirectory with the modelname and timestamp is created')
    parser.add_argument('--test-every-n', type=int, default=5,
        help='Run a full iteration over the test epoch every n epochs')
    parser.add_argument('--alpha', type=float, default=1,
        help='The coefficied in Eq 6 that weights the attention mining loss in relation to the classification loss')
    parser.add_argument('--sigma', type=float, default=0.5,
        help='The threshold value used in Eq 6')
    parser.add_argument('--omega', type=float, default=10,
        help='The scaling value used in Eq 6')
    parser.add_argument('--model', type=str, default='vgg19', choices=gain.available_models,
        help='The name of the underlying model to train')
    parser.add_argument('--pretrain-epochs', type=int, default=100,
        help='The number of epochs to train the network before factoring in the attention map')
    parser.add_argument('--pretrain-threshold', type=float, default=0.95,
        help='The accuracy value to pretrain to before factoring in the attention map loss')

    return parser.parse_args(argv)

def main(argv):
    args = parse_args(argv)

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


if __name__ == '__main__':
    main(sys.argv[1:])
