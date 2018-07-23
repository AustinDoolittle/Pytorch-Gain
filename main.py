#!/usr/bin/env python

from datetime import datetime
import os
import argparse
import sys
import gain
import data
import transform
import torch
import cv2
import time
import models

def set_available_gpus(gpus):
    if isinstance(gpus, list):
        gpu_str = ','.join(gpus)
    else:
        gpu_str = str(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

def train_handler(args):
    if args.gpus:
        set_available_gpus(args.gpus)

    print('Creating Dataset...')
    batch_size_dict = None
    if args.batch_size:
        batch_size_dict = {
            'train': args.batch_size,
            'test': 1
        }

    transformer = None
    if args.transformer:
        transformer = getattr(transform, args.transformer)()
    rds = data.RawDataset(args.dataset_path, output_dims=tuple(args.input_dims),
                          output_channels=args.input_channels, num_workers=args.num_workers,
                          transformer=transformer, batch_size_dict=batch_size_dict)

    output_dir = os.path.join(args.output_dir,
                                rds.name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    heatmap_dir = os.path.join(output_dir, 'heatmaps')
    model_dir = os.path.join(output_dir, 'models')
    gain_args = {
        'gradient_layer_name': args.gradient_layer_name,
        'gpu': bool(args.gpus),
        'heatmap_dir': heatmap_dir,
        'saved_model_dir': model_dir,
        'alpha': args.alpha,
        'omega': args.omega,
        'sigma': args.sigma,
        'batch_norm': not args.no_batch_norm,
    }

    if args.weights_file:
        print('Loading Saved Model from %s...'%args.weights_file)
        if args.input_dims:
            print('WARNING argument input_dims is being ignored in favor of saved model metadata')
        if args.input_channels:
            print('WARNING argument input_channels is being ignored in favor of saved model metadata')
        if args.model_type:
            print('WARNING argument model_type is being ignored in favor of saved model metadata')

        model = gain.AttentionGAIN.load(args.weights_file, **gain_args)
    else:
        print('Creating New Model...')
        gain_args.update({

            'input_channels': rds.output_channels,
            'input_dims': rds.output_dims,
            'labels': rds.labels,
            'model_type': args.model_type
        })
        model = gain.AttentionGAIN(**gain_args)

    print('Starting Training')
    print('=================\n')
    model.train(rds, args.num_epochs, args.serialization_format, pretrain_epochs=args.pretrain_epochs,
                test_every_n_epochs=args.test_every_n_epochs,
                learning_rate=args.learning_rate, num_heatmaps=args.heatmaps_per_test)
    print('\nTraining Complete')
    print('=================')

def infer_handler(args):
    if not args.weights_file:
        raise argparse.ArgumentError('You must specify a weights file when running inference on a file')

    print('Loading model...')
    model = gain.AttentionGAIN.load(args.weights_file, gradient_layer_name=args.gradient_layer_name, batch_norm=not args.no_batch_norm)

    print('Loading data...')
    # load the image file
    image = data.load_image(args.image_path, model.input_dims, model.input_channels)
    image = torch.FloatTensor(image)

    # construct data
    if not args.heatmap_label in model.labels:
        raise argparse.ArgumentError('Label %s not included in model\'s available labels %s'%(args.label, model.labels))

    label_index = model.labels.index(args.heatmap_label)
    label_onehot = torch.zeros(1, len(model.labels))
    label_onehot[0, label_index] = 1

    image = image.expand(label_onehot.size()[0], -1, -1, -1)

    print('Generating heatmap...')
    start_time = time.time()
    output_cl, loss_cl, A_c, heatmap_img = model.generate_heatmap(image, label_onehot)
    time_diff = time.time() - start_time

    print('Inference took %f s'%time_diff)

    if not args.output_dir:
        # display the heatmap
        cv2.imshow('heatmap', heatmap_img)
        cv2.waitKey(1000)
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        out_name = datatime.now().strftime('%Y%m%d_%H%M%S')
        out_name += '_heatmap.png'
        cv2.imwrite(out_name, heatmap_img)

def model_info_handler(args):
    if args.weights_file:
        print('Loading model with weights...')
        # load the meta data too
        model = gain.AttentionGAIN.load(args.weights_file, gradient_layer_name=args.gradient_layer_name, batch_norm=args.batch_norm)
        print(model)
    else:
        print('Loading model...')
        model = models.get_model(args.model_type, 1, batch_norm=not args.no_batch_norm)

        # print every layer in the model
        print('%s Model Layers:'%args.model_type)
        print(models.model_to_str(model))

def parse_args(argv):
    gpu_parent = argparse.ArgumentParser(add_help=False)
    gpu_parent.add_argument('--gpus', type=str, nargs='+',
        help='GPUs to run training on. Exclude for cpu training')

    data_parent = argparse.ArgumentParser(add_help=False)
    data_parent.add_argument('--dataset-path', type=str, required=True,
        help='The path to the dataset, formatted with data in different directories based on label')
    data_parent.add_argument('--num-workers', type=int, default=1,
        help='The number of worker processes to use for loading/transforming data. Note that this spawns this amount of workers for both the test and train dataset.')

    model_parent = argparse.ArgumentParser(add_help=False)
    model_parent.add_argument('--gradient-layer-name', type=str, required=True,
        help='The name of the layer to construct the heatmap from')
    model_parent.add_argument('--model-type', type=str, required=True, choices=models.available_models,
        help='The name of the underlying model to train')
    model_parent.add_argument('--weights-file', type=str,
        help='The full path to the .tar file containing model weights and metadata')
    model_parent.add_argument('--no-batch-norm', action='store_true',
        help='Use batch norm in the custom defined models')

    parser = argparse.ArgumentParser(description='Implementation of GAIN using pytorch')

    subparser = parser.add_subparsers(help='The action to perform')

    train_parser = subparser.add_parser('train',  parents=[gpu_parent, data_parent, model_parent],
        help='Train a new model')
    train_parser.set_defaults(func=train_handler)
    train_parser.add_argument('--learning-rate', type=float, default=0.0005,
        help='Learning rate to plug into the optimizer')
    train_parser.add_argument('--test-every-n-epochs', type=int, default=5,
        help='Run a full iteration over the test epoch every n epochs')
    train_parser.add_argument('--heatmaps-per-test', type=int, default=1,
        help='The number of heatmaps to create for each test')

    train_parser.add_argument('--alpha', type=float, default=1,
        help='The coefficied in Eq 6 that weights the attention mining loss in relation to the classification loss')
    train_parser.add_argument('--sigma', type=float, default=0.4,
        help='The threshold value used in Eq 6. This is a coefficient used as the following: *sigma* * max(*A_c*) ')
    train_parser.add_argument('--omega', type=float, default=100,
        help='The scaling value used in Eq 6')
    train_parser.add_argument('--pretrain-epochs', type=int, default=100,
        help='The number of epochs to train the network before factoring in the attention map')
    train_parser.add_argument('--num-epochs', type=int, default=50,
        help='The number of epochs to run training for')
    train_parser.add_argument('--batch-size', type=int, default=1,
        help='The batch size to use when training')
    train_parser.add_argument('--output-dir', type=str, default='./out',
        help='The output directory for training runs. A subdirectory with the modelname and timestamp is created')
    # TODO dynamically retrieve expected input size???
    train_parser.add_argument('--input-dims', type=int, nargs=2, required=True,
        help='The dimensions to resize inputs to. Keep in mind that some models have a default input size. This is not used if the model is loaded from saved weights.')
    train_parser.add_argument('--input-channels', type=int, required=True,
        help='The number of channels the network should expect as input. This is not used if the model is loaded from saved weights.')
    train_parser.add_argument('--transformer', type=str, choices=transform.available_transformers,
        help='The transformer to use on training data')
    train_parser.add_argument('--serialization-format', type=str, choices=['pytorch', 'onnx'], default='pytorch',
        help='The serialization format to use when saving model checkpoints')

    infer_parser = subparser.add_parser('infer', parents=[gpu_parent, model_parent],
        help='Run inference on a trained model')
    infer_parser.set_defaults(func=infer_handler)
    infer_parser.add_argument('--image-path', type=str, required=True,
        help='The path to the image that you would like to classify')
    infer_parser.add_argument('--heatmap-label', type=str, required=True,
        help='If this is set, a heatmap is only generated for this label. Otherwise, a heatmap is generated for all labels')
    infer_parser.add_argument('--output-dir', type=str,
        help='The directory to save heatmap outputs')

    model_info_parser = subparser.add_parser('model', parents=[model_parent],
        help='Utility to print information about a model for easier layer selection')
    model_info_parser.set_defaults(func=model_info_handler)

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    args.func(args)
