import torch
from datetime import datetime
import numpy as np
import os
import torchvision
import cv2
import numpy
import argparse
import sys
import random
import re

model_file_reg = re.compile(r'(?P<model_type>[a-zA-Z0-9-_]+)__(?P<epoch>\d+)__(?P<tag>[a-zA-Z0-9-_]+)\.pt')
available_models = dir(torchvision.models)


def scalar(tensor):
    return tensor.data.cpu().numpy()[0]

def set_available_gpus(gpus):
    if isinstance(gpus, list):
        gpu_str = ','.join(gpus)
    else:
        gpu_str = str(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

def load_model(model_type, output_length):
    model = getattr(torchvision.models, model_type)
    return model(pretrained=False, num_classes=output_length)

class RawDataset:
    def __init__(self, root_dir, ds_split=0.8, include_exts=['.jpg', '.png', '.jpeg'], transformer=None, out_dims=(224, 224)):
        self._ds_split = ds_split
        self.root_dir = root_dir
        self.include_exts = include_exts
        self.out_dims = out_dims
        self.transformer = transformer

        self.datasets = self._load_datasets()

    def _load_datasets(self):
        # iterate over the immediate child directories and store paths to images
        image_dict = {}
        for p in os.listdir(self.root_dir):
            full_dir = os.path.join(self.root_dir, p)
            if not os.path.isdir(full_dir):
                continue

            for f in os.listdir(full_dir):
                if os.path.splitext(f)[1] in self.include_exts:
                    if not p in image_dict:
                        image_dict[p] = []

                    image_dict[p].append(os.path.join(self.root_dir, p, f))
        self.labels = image_dict.keys()

        # split the train and test datasets equally among labels
        filesets = {'test': [], 'train': []}
        for k, filenames in image_dict.items():
            random.shuffle(filenames)
            split_index = int(len(filenames) * self._ds_split)
            filesets['train'] += [(f, k) for f in filenames[:split_index]]
            filesets['test'] += [(f, k) for f in filenames[split_index:]]

        ret_dict = {}

        # create image datasets for train and test
        for k in filesets:
            ds_args = {
                'out_dims': self.out_dims,
            }
            if k == 'train':
                ds_args['transformer'] = self.transformer

            ds = ImageDataset(filesets['train'], self.labels, **ds_args)
            ret_dict[k] = torch.utils.data.DataLoader(ds, shuffle=True, num_workers=2)

        return ret_dict


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, fileset, labels, transformer=None, out_dims=(256,256)):
        self.out_dims = out_dims
        self.transformer = transformer
        self._fileset = fileset
        self.labels = labels

    def __len__(self):
        return len(self._fileset)

    def __getitem__(self, idx):
        # decode and resize
        decoded_img = cv2.imread(self._fileset[idx][0])
        decoded_img = cv2.resize(decoded_img, self.out_dims)

        # pytorch is [C, H, W]
        decoded_img = decoded_img.transpose((2, 0, 1))
        decoded_img = decoded_img.astype(np.float32)
        decoded_img = decoded_img / 255.0

        label_index = self.labels.index(self._fileset[idx][1])

        ret_dict = {
            'image': decoded_img,
            'label/idx': label_index,
            'label/name': self._fileset[idx][1]
        }

        # transform 'em if we got 'em
        if self.transformer:
            ret_dict = self.transformer(ret_dict)

        return ret_dict


class AttentionGAIN:
    def __init__(self, rds, model_type, gradient_layer_name, learning_rate=0.005, alpha=1, omega=10, sigma=0.5,
                gpu=False, heatmap_dir=None, saved_model_dir=None, load_weights=False):
        self.gpu = gpu

        # TODO open this up to other models
        self.model_type = model_type

        self.model = load_model(self.model_type, len(rds.labels))

        if self.gpu:
            self.model = self.model.cuda()

        self._register_hooks(gradient_layer_name)

        self.rds = rds
        self.loss_cl = torch.nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.heatmap_dir = os.path.abspath(heatmap_dir)
        self.saved_model_dir = os.path.abspath(saved_model_dir)
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha

    @staticmethod
    def _create_saved_model_name(model_type, epoch, tag):
        return '%s_%i_%s.pt'%(model_type, epoch, tag)

    @staticmethod
    def _parse_saved_model_name(filename):
        result = model_file_reg.match(filename)
        if not result:
            raise ValueError('Could not parse model information from filename %s'%filename)

        return result.group('model_type'), int(result.group('epoch')), result.group('tag')


    def _register_hooks(self, layer_name):
        # this wires up a hook that stores both the activation and gradient of the conv layer we are interested in
        def forward_hook(module, input_, output_):
            self._last_activation = output_

        def backward_hook(module, grad_in, grad_out):
            self._last_grad = grad_out[0]

        # locate the layer that we are concerned about
        gradient_layer_found = False
        for idx, m in self.model.named_modules():
            if idx == layer_name:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model'%layer_name)

    def _to_one_hot(self, in_tensor):
        # utility function for turning list of integers to one_hot encoded arrays
        in_tensor = in_tensor.view(1,-1)
        one_hot = torch.zeros(in_tensor.size()[0], len(self.rds.labels))

        if self.gpu:
            one_hot = one_hot.cuda()

        one_hot = one_hot.scatter_(1, in_tensor, 1.0)

        return one_hot

    def _convert_data_and_label(self, data, label):
        # converts our data and label over to variables, gpu optional
        if self.gpu:
            data = data.cuda()
            label = label.cuda()

        data = torch.autograd.Variable(data)
        label_one_hot = self._to_one_hot(label)
        label_one_hot = torch.autograd.Variable(label_one_hot)

        return data, label_one_hot

    def _maybe_save_model(self, epoch, tag='default', save_count=1):
        # TODO if a different save count but same tag is used in different circumstances, this will have
        # undefined behavior (we only delete one file if there are too many, but we should be trimming to *save_count*
        if self.saved_model_dir is None:
            return


        if not os.path.exists(self.saved_model_dir):
            try:
                os.makedirs(self.saved_model_dir)
            except OSError as e:
                print('WARNING there was an error while creating directory %s: %s'%(str(self.saved_model_dir), str(e)))
                return

        min_epoch = epoch
        delete_model_path = None
        num_models = 0
        # store the model for later deletion
        for p in os.listdir(self.saved_model_dir):
            if not os.path.splitext(p)[-1] == '.pt':
                # Don't try with this one
                continue

            try:
                _, temp_epoch, temp_tag = self._parse_saved_model_name(p)
            except ValueError as e:
                print('WARNING error while parsing saved model filename: %s'%str(e))
                continue

            if temp_tag != tag:
                continue

            num_models += 1
            if temp_epoch < min_epoch:
                delete_model_path = os.path.join(self.saved_model_dir, p)
                min_epoch = temp_epoch

        # if we are less that the max saved model count, then don't worry about it
        if num_models < save_count:
            delete_model_path = None

        # save the current model
        model_name = self._create_saved_model_name(self.model_type, epoch, tag)
        model_name = os.path.join(self.saved_model_dir, model_name)
        try:

            torch.save(self.model.state_dict(), model_name)
        except OSError as e:
            print('WARNING there was an error while saving model: %s'%str(e))
            return

        # delete our extra model
        if delete_model_path:
            try:
                os.delete
            except OSError as e:
                print('WARNING there was an error while trying to remove file %s: %s'% (delete_model_path, e))


    def _maybe_save_heatmap(self, image, heatmap, epoch):
        if self.heatmap_dir is None:
            return

        # get the min and max values once to be used with scaling
        min_val = heatmap.min()
        max_val = heatmap.max()

        # Scale the heatmap in range 0-255
        # CAUTION: gross
        heatmap = (255 * (heatmap - min_val)) / (max_val - min_val + 1e-5)
        heatmap = heatmap.data.cpu().numpy().astype(np.uint8).transpose((1,2,0))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Scale the image as well
        scaled_image = image * 255.0
        scaled_image = scaled_image.cpu().numpy().astype(np.uint8).transpose((1,2,0))

        if scaled_image.shape[2] == 1:
            scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2RGB)

        # generate the heatmap
        heatmap_image = cv2.addWeighted(scaled_image, 0.7, heatmap, 0.3, 0)

        # write it to a file
        if not os.path.exists(self.heatmap_dir):
            os.makedirs(self.heatmap_dir)

        out_file = os.path.join(self.heatmap_dir, 'epoch_%i.png'%epoch)
        cv2.imwrite(os.path.join(self.heatmap_dir, 'epoch_%i.png'%epoch), heatmap_image)
        print('TEST Heatmap written to %s'%out_file)

    def forward(self, data, label):
        data, label_one_hot = self._convert_data_and_label(data, label)

        return self._forward(data, label_one_hot)

    def train(self, epochs, pretrain_epochs=10, pretrain_threshold=0.95, test_every_n_epochs=5):
        last_acc = 0
        max_acc = 0
        pretrain_finished = False
        for i in range(epochs):
            pretrain_finished = pretrain_finished or \
                                i > pretrain_epochs or \
                                last_acc >= pretrain_threshold
            loss_cl_sum = 0
            loss_am_sum = 0
            acc_cl_sum = 0
            total_loss_sum = 0
            # train
            for sample in self.rds.datasets['train']:
                total_loss, loss_cl, loss_am, acc_cl, A_c = self.forward(sample['image'], sample['label/idx'])
                total_loss_sum += scalar(total_loss)
                loss_cl_sum += scalar(loss_cl)
                loss_am_sum += scalar(loss_am)
                acc_cl_sum += scalar(acc_cl)

                # Backprop that thang up
                loss_cl_tag = 'Loss_CL'
                loss_total_tag = 'Loss_Total'
                if pretrain_finished:
                    loss_total_tag = '<' + loss_total_tag + '>'
                    total_loss.backward()
                else:
                    loss_cl_tag = '<' + loss_cl_tag + '>'
                    loss_cl.backward()

                self.opt.step()
            train_size = len(self.rds.datasets['train'])
            last_acc = acc_cl_sum / train_size
            print('[Epoch %i] %s: %f, Loss_AM: %f, %s: %f, Accuracy_CL: %f%%'%
                    ((i+1), loss_cl_tag, loss_cl_sum / train_size, loss_am_sum / train_size, loss_total_tag, total_loss_sum / train_size, last_acc * 100.0))


            if (i + 1) % test_every_n_epochs == 0:
                # test
                loss_cl_sum = 0
                loss_am_sum = 0
                acc_cl_sum = 0
                total_loss_sum = 0
                for sample in self.rds.datasets['test']:
                    # test
                    total_loss, loss_cl, loss_am, acc_cl, A_c = self.forward(sample['image'], sample['label/idx'])

                    total_loss_sum += scalar(total_loss)
                    loss_cl_sum += scalar(loss_cl)
                    loss_am_sum += scalar(loss_am)
                    acc_cl_sum += scalar(acc_cl)

                test_size = len(self.rds.datasets['test'])
                avg_acc = acc_cl_sum / test_size
                if avg_acc > max_acc:
                    self._maybe_save_model(i+1)

                print('TEST Loss_CL: %f, Loss_AM: %f, Loss_Total: %f, Accuracy_CL: %f%%'%
                    (loss_cl_sum / test_size, loss_am_sum / test_size, total_loss_sum / test_size, avg_acc * 100.0))

                self._maybe_save_heatmap(sample['image'][0], A_c[0], i+1)

    def _forward(self, data, label):
        # self._clear_gradients()
        output_cl = self.model(data)

        output_cl.backward(gradient=label)
        self.model.zero_grad()

        # Eq 1
        w_c = self._last_grad.mean(dim=(1), keepdim=True)

        # Eq 2
        A_c = self._last_activation * w_c
        A_c = A_c.sum(dim=(1), keepdim=True)
        A_c = torch.nn.functional.relu(A_c)

        # Eq 4
        T_A_c = torch.sigmoid(self.omega * (A_c - self.sigma))

        # resize our maps to be the size of the input image
        A_c_resized = torch.nn.functional.upsample(A_c, size=data.size()[2:], mode='bilinear')
        T_A_c_resized = torch.nn.functional.upsample(T_A_c, size=data.size()[2:], mode='bilinear')

        # Eq 3
        I_star = data - (T_A_c_resized * data)

        output_am = self.model(I_star)

        # Eq 5
        loss_am = torch.nn.functional.softmax(output_am, dim=1) * label

        tensor_module = torch
        if self.gpu:
            tensor_module = torch.cuda

        loss_am = loss_am.sum() / label.sum().type(tensor_module.FloatTensor)

        loss_cl = self.loss_cl(output_cl, label)

        # Eq 6
        total_loss = loss_cl + self.alpha*loss_am

        cl_acc = output_cl.max(dim=1)[1] == label.max(dim=1)[1]
        cl_acc = cl_acc.type(tensor_module.FloatTensor).mean()

        return total_loss, loss_cl, loss_am, cl_acc, A_c_resized


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--gpus', type=str, nargs='+')
    parser.add_argument('--gradient-layer-name', type=str, default='features.34')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='./out')
    parser.add_argument('--test-every-n', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--omega', type=float, default=10)
    parser.add_argument('--model', type=str, default='vgg19', choices=available_models)
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--pretrain-threshold', type=float, default=0.95)

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
    rds = RawDataset(args.dataset_path)

    print('Creating Model...')
    model = AttentionGAIN(rds, args.model, args.gradient_layer_name, learning_rate=args.learning_rate,
                          gpu=bool(args.gpus), heatmap_dir=heatmap_dir, saved_model_dir=model_dir, alpha=args.alpha)

    print('Starting Training')
    print('=================\n')
    model.train(args.num_epochs, pretrain_epochs=args.pretrain_epochs,
                pretrain_threshold=args.pretrain_threshold, test_every_n_epochs=args.test_every_n)
    print('\nTraining Complete')
    print('=================')


if __name__ == '__main__':
    main(sys.argv[1:])
