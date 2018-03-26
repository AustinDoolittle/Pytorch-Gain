import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import os
import torchvision
import cv2
import re
import io
import json
import math

model_file_reg = re.compile(r'saved_model_(?P<epoch>\d+)_(?P<tag>[a-zA-Z0-9-_]+)\.model')
available_models = [i for i in dir(torchvision.models) if not i.startswith('_')]

def scalar(tensor):
    return tensor.data.cpu().numpy()[0]

def get_model(model_type, output_length):
    model = getattr(torchvision.models, model_type)
    return model(pretrained=False, num_classes=output_length)

def tile_images(images, num_wide=5):
    # dummy check
    if len(images) == 1:
        return images[0]

    shape = images[0].shape
    num_high = math.ceil(len(images) / float(num_wide))
    num_high = int(num_high)

    if len(images) < num_wide:
        num_wide = len(images)

    output_img = np.zeros((num_high * shape[0], num_wide * shape[1], shape[2]), dtype=np.uint8)
    for i in range(len(images)):
        curr_image = images[i]
        if curr_image.shape != shape:
            print('WARNING image shape is not consistent, images are all resized to %s'%str(shape))
            curr_image = cv2.resize(curr_image, shape[:2])

        y = i % num_wide * shape[0]
        x = int(i / num_wide) * shape[1]
        output_img[x:x + shape[0], y:y + shape[1], :] = curr_image
    return output_img


class AttentionGAIN:
    def __init__(self, model_type=None, gradient_layer_name=None, weights=None, heatmap_dir=None,
                 saved_model_dir=None, epoch=0, gpu=False, alpha=1, omega=10, sigma=0.5, labels=None,
                 input_channels=None, input_dims=None):
        # validation
        if not model_type:
            raise ValueError('Missing required argument model_type')
        if not gradient_layer_name:
            raise ValueError('Missing required argument gradient_layer_name')
        if not input_channels:
            raise ValueError('Missing required argument input_channels')
        if not input_dims:
            raise ValueError('Missing required argument input_dims')

        # set gpu options
        self.gpu = gpu

        # define model
        self.model_type = model_type
        self.model = get_model(self.model_type, len(labels))
        if weights:
            self.model.load_state_dict(weights)
            self.epoch = epoch
        elif epoch > 0:
            raise ValueError('epoch_offset > 0, but no weights were supplied')

        if self.gpu:
            self.model = self.model.cuda()
            self.tensor_source = torch.cuda
        else:
            self.tensor_source = torch

        # wire up our hooks for heatmap creation
        self._register_hooks(gradient_layer_name)

        # create loss function
        # TODO make this configurable
        self.loss_cl = torch.nn.BCEWithLogitsLoss()

        # output directory setup
        self.heatmap_dir = heatmap_dir
        if self.heatmap_dir:
            self.heatmap_dir = os.path.abspath(self.heatmap_dir)

        self.saved_model_dir = saved_model_dir
        if self.saved_model_dir:
            self.saved_model_dir = os.path.abspath(saved_model_dir)

        # misc. parameters
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha
        self.labels = labels
        self.input_channels = input_channels
        self.input_dims = input_dims
        self.epoch = epoch

    @staticmethod
    def load(model_path, **kwargs):
        model_dict= torch.load(model_path)
        kwargs.update(model_dict['meta'])
        return AttentionGAIN(weights=model_dict['state_dict'], **kwargs)

    @staticmethod
    def _parse_saved_model_name(model_name):
        result = model_file_reg.match(model_name)
        if not result:
            raise ValueError('Could not parse tag from model name %s'%model_name)

        return result.group('epoch'), result.group('tag')

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

    def __str__(self):
        meta_dict = self._get_meta_dict()
        ret_str = 'Metadata:'
        for k,v in meta_dict.items():
            ret_str += '\n\t%s: %s'%(str(k), str(v))
        ret_str += '\n'

        ret_str += 'Layers:'
        for idx, m in self.model.named_modules():
            if not '.' in str(idx):
                continue
            ret_str += '\n%s: %s'%(str(idx), str(m))

        return ret_str

    def _convert_data_and_label(self, data, label):
        # converts our data and label over to variables, gpu optional
        if self.gpu:
            data = data.cuda()
            label = label.cuda()

        data = torch.autograd.Variable(data)
        label = torch.autograd.Variable(label)

        return data, label

    def _get_meta_dict(self):
        ret_dict = {
                'input_dims': self.input_dims,
                'input_channels': self.input_channels,
                'model_type': self.model_type,
                'labels': self.labels,
                'epoch': self.epoch
            }
        return ret_dict

    def _maybe_save_model(self, tag='default', save_count=1):
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

        max_epoch = self.epoch
        delete_model_path = None
        num_models = 0
        # store the model for later deletion
        for p in os.listdir(self.saved_model_dir):
            if not os.path.splitext(p)[-1] == '.model':
                continue

            try:
                temp_epoch, temp_tag = self._parse_saved_model_name(p)
            except ValueError as e:
                print('WARNING error while parsing saved model filename: %s'%str(e))
                continue
            if temp_tag != tag:
                continue

            temp_epoch = int(temp_epoch)
            num_models += 1
            full_model_path = os.path.join(self.saved_model_dir, p)
            if temp_epoch < max_epoch:
                delete_model_path = os.path.join(self.saved_model_dir, p)
                max_epoch = temp_epoch

        # if we are less that the max saved model count, then don't worry about it
        if num_models < save_count:
            delete_model_path = None

        # save the current model
        saved_model_filename = os.path.join(self.saved_model_dir, 'saved_model_%i_%s.model'%(self.epoch, tag))

        save_dict = {
            'state_dict': self.model.state_dict(),
            'meta': self._get_meta_dict()
        }

        try:
            torch.save(save_dict, saved_model_filename)
            print('MODEL saved to %s'%saved_model_filename)
        except OSError as e:
            print('WARNING there was an error while saving model: %s'%str(e))
            return

        # delete our extra model
        if delete_model_path:
            try:
                os.remove(delete_model_path)
            except OSError as e:
                print('WARNING there was an error while trying to remove file %s: %s'% (delete_model_path, e))

    def _maybe_save_heatmap(self, images, labels, heatmaps, epoch, heatmap_nbr):
        if self.heatmap_dir is None:
            return

        heatmap_image = self._combine_heatmaps(images, labels, heatmaps)

        # write it to a file
        if not os.path.exists(self.heatmap_dir):
            os.makedirs(self.heatmap_dir)

        out_file = os.path.join(self.heatmap_dir, 'epoch_%i_%i.png'%(epoch, heatmap_nbr))
        cv2.imwrite(out_file, heatmap_image)

        print('HEATMAP saved to %s'%out_file)

    def _combine_heatmaps(self, data, labels, A_c):
        heatmaps = []
        if len(labels.size()) == 2:
            labels = labels.max(dim=1)[1]

        labels = labels.cpu().numpy()

        for i in range(data.shape[0]):
            heatmaps.append(self._combine_heatmap_with_image(data[i], A_c[i], self.labels[labels[i]]))

        out_heatmap = tile_images(heatmaps)
        return out_heatmap

    @staticmethod
    def _combine_heatmap_with_image(image, heatmap, label_name):
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

        # superimpose label_name
        img_height = heatmap_image.shape[0]
        heatmap_image = cv2.putText(heatmap_image, label_name,
                                    (10, img_height - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    (255, 255, 255),
                                    1,
                                    cv2.LINE_AA)
        return heatmap_image

    def generate_heatmap(self, data, label, width=3):
        data_var, label_var = self._convert_data_and_label(data, label)
        output_cl, loss_cl, A_c = self._attention_map_forward(data_var, label_var)
        heatmap = self._combine_heatmaps(data, label, A_c)
        return output_cl, loss_cl, A_c, heatmap

    def forward(self, data, label):
        data, label = self._convert_data_and_label(data, label)
        return self._forward(data, label)

    def _check_dataset_compatability(self, rds):
        if rds.output_dims != self.input_dims:
            raise ValueError('Dataset outputs images with dimension %s, model expects %s'%
                                (str(rds.output_dims), str(self.input_dims)))
        elif rds.output_channels != self.input_channels:
            raise ValueError('Dataset outputs images with channel_count %i, model expects %i'%
                                (rds.output_channels, self.input_channels))
        elif rds.labels != self.labels:
            raise ValueError('Dataset has labels %s, model has labels %s'%
                                (str(rds.labels), str(self.labels)))

    def train(self, rds, epochs, pretrain_epochs=10, learning_rate=1e-5, pretrain_threshold=0.95, test_every_n_epochs=5, num_heatmaps=1, heatmap_all_labels=False):
        # TODO dynamic optimizer selection
        self._check_dataset_compatability(rds)

        last_acc = 0
        max_acc = 0
        pretrain_finished = False

        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for i in range(self.epoch, epochs, 1):
            self.epoch = i
            pretrain_finished = pretrain_finished or \
                                i > pretrain_epochs or \
                                last_acc >= pretrain_threshold
            loss_cl_sum = 0
            loss_am_sum = 0
            acc_cl_sum = 0
            total_loss_sum = 0
            # train
            for sample in rds.datasets['train']:
                total_loss, loss_cl, loss_am, probs, acc_cl, A_c = self.forward(sample['image'], sample['label/onehot'])
                total_loss_sum += scalar(total_loss)
                loss_cl_sum += scalar(loss_cl)
                loss_am_sum += scalar(loss_am)
                acc_cl_sum += scalar(acc_cl)

                # Backprop selectively based on pretraining/training
                if pretrain_finished:
                    print_prefix = 'TRAIN'
                    total_loss.backward()
                else:
                    print_prefix = 'PRETRAIN'
                    loss_cl.backward()

                opt.step()
            train_size = len(rds.datasets['train'])
            last_acc = acc_cl_sum / train_size
            print('%s Epoch %i, Loss_CL: %f, Loss_AM: %f, Loss Total: %f, Accuracy_CL: %f%%'%
                    (print_prefix, (i+1), loss_cl_sum / train_size, loss_am_sum / train_size,
                     total_loss_sum / train_size, last_acc * 100.0))


            if (i + 1) % test_every_n_epochs == 0:
                # test
                loss_cl_sum = 0
                loss_am_sum = 0
                acc_cl_sum = 0
                total_loss_sum = 0
                heatmap_count = 0
                for sample in rds.datasets['test']:
                    data = sample['image']
                    label_onehot = sample['label/onehot']
                    label = sample['label/idx']
                    save_heatmap = heatmap_count < num_heatmaps

                    if save_heatmap and heatmap_all_labels:
                        label_onehot = torch.eye(len(self.labels))
                        label = torch.arange(0, len(self.labels), out=torch.LongTensor())
                        data = data.expand(len(self.labels), -1, -1, -1)

                    # test
                    total_loss, loss_cl, loss_am, prob, acc_cl, A_c = self.forward(data, label_onehot)

                    if save_heatmap:
                        self._maybe_save_heatmap(data, label, A_c, i+1, heatmap_count)
                        heatmap_count += 1

                    total_loss_sum += scalar(total_loss)
                    loss_cl_sum += scalar(loss_cl)
                    loss_am_sum += scalar(loss_am)
                    acc_cl_sum += scalar(acc_cl)

                test_size = len(rds.datasets['test'])
                avg_acc = acc_cl_sum / test_size
                print('TEST Loss_CL: %f, Loss_AM: %f, Loss_Total: %f, Accuracy_CL: %f%%'%
                    (loss_cl_sum / test_size, loss_am_sum / test_size, total_loss_sum / test_size, avg_acc * 100.0))

                if avg_acc > max_acc:
                    self._maybe_save_model()
                    max_acc = avg_acc

    def _attention_map_forward(self, data, label):
        output_cl = self.model(data)
        output_cl.backward(gradient=label * output_cl)

        self.model.zero_grad()

        # Eq 1
        w_c = self._last_grad.mean(dim=(1), keepdim=True)

        # paper calls this convolution, I think it's easier to think of it as a sum of products
        # Eq 2
        A_c = self._last_activation * w_c
        A_c = A_c.sum(dim=(1), keepdim=True)
        A_c = F.relu(A_c)
        A_c = F.upsample(A_c, size=data.size()[2:], mode='bilinear')

        loss_cl = self.loss_cl(output_cl, label)

        return output_cl, loss_cl, A_c

    def _forward(self, data, label):
        output_cl, loss_cl, A_c = self._attention_map_forward(data, label)
        output_cl_softmax = F.softmax(output_cl, dim=1)

        # Eq 4
        T_A_c = torch.sigmoid(self.omega * (A_c - A_c.mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)))
        # T_A_c = torch.sigmoid(self.omega * (A_c - self.sigma))

        # Eq 3
        I_star = data - (T_A_c * data)

        output_am = self.model(I_star)

        # Eq 5
        loss_am = F.softmax(output_am, dim=1) * label

        loss_am = loss_am.sum() / label.sum().type(self.tensor_source.FloatTensor)

        # Eq 6
        total_loss = loss_cl + self.alpha*loss_am

        cl_acc = output_cl_softmax.max(dim=1)[1] == label.max(dim=1)[1]
        cl_acc = cl_acc.type(self.tensor_source.FloatTensor).mean()

        return total_loss, loss_cl, loss_am, output_cl_softmax, cl_acc, A_c


