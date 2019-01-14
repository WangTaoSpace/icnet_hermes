#!/bin/env python
# -*- encoding:utf-8 -*-
import paddle.fluid as fluid
import numpy as np
import sys
import os
import cv2
import paddle.dataset as dataset
from paddle.fluid import core
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
from PIL import Image
import paddle
# default parameters
trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
training_role = os.getenv("TRAINING_ROLE", "TRAINER")
port = os.getenv("PADDLE_PORT", "6174")
pserver_ips = os.getenv("PADDLE_PSERVERS")
current_endpoint = str(os.getenv("POD_IP")) + ":" + port

if 'ce_mode' in os.environ:  
    np.random.seed(10)
    fluid.default_startup_program().random_seed = 90
# dataset reader
# this method is for recordio format
# if you have other formats, try to modify it
IMG_MEAN = 52
TRAIN_LIST = []
NUM_CLASSES = 2
#TRAIN_DATA_SHAPE = (3, 1710, 3384)
TRAIN_DATA_SHAPE = (1,320, 640)

#IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)  # sub avg, get model fast

train_data_dirs = {}
label_dirs = {}


# label_dict = {0:0, 249:0, 255:0,
#               200:1, 204:1, 213:1, 209:1, 206:1, 207:1,
#               201:2, 203:2, 211:2, 208:2,
#               216:3, 217:3, 215:3,
#               218:4, 219:4,
#               210:5, 232:5,
#               214:6,
#               202:7, 220:7, 221:7, 222:7, 231:7, 224:7, 225:7, 226:7, 230:7, 228:7, 229:7, 233:7,
#               205:8, 212:8, 227:8, 223:8, 250:8}
#
label_dict = {0:0, 249:0, 255:0,
              200:1, 204:1, 213:1, 209:1, 206:1, 207:1,
              201:1, 203:1, 211:1, 208:1,
              216:1, 217:1, 215:1,
              218:1, 219:1,
              210:1, 232:1,
              214:1,
              202:1, 220:1, 221:1, 222:1, 231:1, 224:1, 225:1, 226:1, 230:1, 228:1, 229:1, 233:1,
              205:1, 212:1, 227:1, 223:1, 250:1}
class DataGenerater:
    def __init__(self, data_list, flip=False, scaling= False ,corping= False):
        self.flip = flip
        self.scaling = scaling
        self.corping = corping
        self.image_label = []  # train is (image,label),test is (image)
        for image_file, label_file in data_list:
            self.image_label.append((image_file, label_file))

    def create_train_reader(self):
        """
        Create a reader for train dataset.
        """

        def reader():
            np.random.shuffle(self.image_label)   #
            for image, label in self.image_label:
                image, label_sub1, label_sub2, label_sub4 = self.process_train_data(image,label)
                yield self.mask(
                    np.array(image).astype("float32"),
                    np.array(label_sub1).astype("float32"),
                    np.array(label_sub2).astype("float32"),
                    np.array(label_sub4).astype("float32"))

        return reader

    def process_train_data(self, image, label):
        """
        Process training data.
        """
        image, label = self.load(image, label)
        if self.flip:
            image, label = self.random_flip(image, label)
        if self.scaling:
            image, label = self.random_scaling(image, label)
        if self.corping:
            image, label = self.resize(image, label, out_size=TRAIN_DATA_SHAPE[1:])
        #print("label",label.shape)
        label_sub1 = self.scale_label(label, factor=4)
        label_sub2 = self.scale_label(label, factor=8)
        label_sub4 = self.scale_label(label, factor=16)
        return image, label_sub1, label_sub2, label_sub4

    def load(self, image, label):
        """
        Load image from file.
        """
        image = np.array(Image.open(image).convert("L"))
        image = cv2.resize(
            image, (TRAIN_DATA_SHAPE[2], TRAIN_DATA_SHAPE[1]), interpolation=cv2.INTER_NEAREST)
        image = image[np.newaxis,:,:] # translate  chw
        image -= IMG_MEAN
        label = np.array(Image.open(label))
        label = cv2.resize(
            label, (TRAIN_DATA_SHAPE[2], TRAIN_DATA_SHAPE[1]), interpolation=cv2.INTER_NEAREST)
        [rows, cols] = label.shape
        for i in range(rows):
            for j in range(cols):
                label[i,j] = label_dict.get(label[i,j],0)    
        return image, label

    def random_flip(self, image, label):
        """
        Flip image and label randomly.
        """
        r = np.random.rand(1)
        if r > 0.5:
            image = dataset.image.left_right_flip(image, is_color=True)
            label = dataset.image.left_right_flip(label, is_color=False)
        return image, label

    def random_scaling(self, image, label):
        """
        Scale image and label randomly.
        """
        scale = np.random.uniform(0.5, 2.0, 1)[0]
        h_new = int(image.shape[0] * scale)
        w_new = int(image.shape[1] * scale)
        image = cv2.resize(image, (w_new, h_new))
        label = cv2.resize(
            label, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        #print("random_scaling",set(label.flatten().tolist()))
        return image, label

    def padding_as(self, image, h, w, is_color):
        """
        Padding image.
        """
        pad_h = max(image.shape[0], h) - image.shape[0]
        pad_w = max(image.shape[1], w) - image.shape[1]
        if is_color:
            return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        else:
            return np.pad(image, ((0, pad_h), (0, pad_w)), 'constant')

    def resize(self, image, label, out_size):
        """
        Resize image and label by padding or cropping.
        """
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        combined = np.concatenate((image, label), axis=2)
        combined = self.padding_as(
            combined, out_size[0], out_size[1], is_color=True)
        combined = dataset.image.random_crop(
            combined, out_size[0], is_color=True)
        image = combined[:, :, 0:3]
        label = combined[:, :, 3:4]
        #print("resize",set(label.flatten().tolist()))
        return image, label

    def scale_label(self, label, factor):
        """
        Scale label according to factor.
        """
        h = label.shape[0] // factor
        w = label.shape[1] // factor
        label= cv2.resize(
            label, (w, h), interpolation=cv2.INTER_NEAREST)
        return label


    def mask(self, image, label0, label1, label2):
        """
        Get mask for valid pixels.
        """
        # mask_sub1 = np.where((label0 > 0).flatten())[0].astype("int32")
        # mask_sub2 = np.where((label1 > 0).flatten())[0].astype("int32")
        # mask_sub4 = np.where((label2 > 0).flatten())[0].astype("int32")
        # return image.astype(
        #     "float32"), label0, mask_sub1, label1, mask_sub2, label2, mask_sub4
        return image,label0,label1,label2


##################################################################
class icnet:
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, relu=False, padding="VALID", biased=False, name=None):
        act = None
        tmp = input
        if relu:
            act = "relu"
        if padding == "SAME":
            padding_h = max(k_h - s_h, 0)  #
            padding_w = max(k_w - s_w, 0)
            padding_top = padding_h // 2
            padding_left = padding_w // 2
            padding_bottom = padding_h - padding_top
            padding_right = padding_w - padding_left
            padding = [
                0, 0, 0, 0, padding_top, padding_bottom, padding_left, padding_right
            ]
            tmp = fluid.layers.pad(tmp, padding)
        tmp = fluid.layers.conv2d(
            tmp,
            num_filters=c_o,
            filter_size=[k_h, k_w],
            stride=[s_h, s_w],
            groups=1,
            act=act,
            bias_attr=biased,
            use_cudnn=False,
            name=name)
        return tmp

    def atrous_conv(self, input, k_h, k_w, c_o, dilation, relu=False, padding="VALID", biased=False, name=None):
        act = None
        if relu:
            act = "relu"
        tmp = input
        tmp = fluid.layers.conv2d(
            input,
            num_filters=c_o,
            filter_size=[k_h, k_w],  #
            dilation=dilation,
            groups=1,
            act=act,
            bias_attr=biased,
            use_cudnn=False,
            name=name)
        return tmp

    def zero_padding(self, input, padding):  
        return fluid.layers.pad(input,
                                [0, 0, 0, 0, padding, padding, padding, padding])

    def bn(self, input, relu=False, name=None, is_test=False):
        act = None
        if relu:
            act = 'relu'
        name = input.name.split(".")[0] + "_bn"
        tmp = fluid.layers.batch_norm(  
            input, act=act, momentum=0.95, epsilon=1e-5, name=name)
        return tmp

    def avg_pool(self, input, k_h, k_w, s_h, s_w, name=None, padding=0):
        temp = fluid.layers.pool2d(
            input,
            pool_size=[k_h, k_w],
            pool_type="avg",
            pool_stride=[s_h, s_w],
            pool_padding=padding,
            name=name)
        return temp

    def max_pool(self, input, k_h, k_w, s_h, s_w, name=None, padding=0):
        temp = fluid.layers.pool2d(
            input,
            pool_size=[k_h, k_w],
            pool_type="max",
            pool_stride=[s_h, s_w],
            pool_padding=padding,
            name=name)
        return temp

    def interp(self, input, out_shape):  # 
        out_shape = list(out_shape.astype("int32"))
        return fluid.layers.resize_bilinear(input, out_shape=out_shape)

    def dilation_convs(self, input):  
        tmp = self.res_block(input, filter_num=256, padding=1, name="conv3_2")
        tmp = self.res_block(tmp, filter_num=256, padding=1, name="conv3_3")
        tmp = self.res_block(tmp, filter_num=256, padding=1, name="conv3_4")

        tmp = self.proj_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_1")
        tmp = self.res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_2")
        tmp = self.res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_3")
        tmp = self.res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_4")
        tmp = self.res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_5")
        tmp = self.res_block(tmp, filter_num=512, padding=2, dilation=2, name="conv4_6")

        tmp = self.proj_block(
            tmp, filter_num=1024, padding=4, dilation=4, name="conv5_1")
        tmp = self.res_block(tmp, filter_num=1024, padding=4, dilation=4, name="conv5_2")
        tmp = self.res_block(tmp, filter_num=1024, padding=4, dilation=4, name="conv5_3")
        return tmp

    def pyramis_pooling(self, input, input_shape):  #
        shape = np.ceil(input_shape // 32).astype("int32")
        h, w = shape
        pool1 = self.avg_pool(input, h, w, h, w)
        pool1_interp = self.interp(pool1, shape)
        pool2 = self.avg_pool(input, h // 2, w // 2, h // 2, w // 2)
        pool2_interp = self.interp(pool2, shape)
        pool3 = self.avg_pool(input, h // 3, w // 3, h // 3, w // 3)
        pool3_interp = self.interp(pool3, shape)
        pool4 = self.avg_pool(input, h // 4, w // 4, h // 4, w // 4)
        pool4_interp = self.interp(pool4, shape)
        conv5_3_sum = input + pool4_interp + pool3_interp + pool2_interp + pool1_interp
        return conv5_3_sum

    def shared_convs(self, image):
        tmp = self.conv(image, 3, 3, 32, 2, 2, padding='SAME', name="conv1_1_3_3_s2")
        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, 3, 3, 32, 1, 1, padding='SAME', name="conv1_2_3_3")
        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, 3, 3, 64, 1, 1, padding='SAME', name="conv1_3_3_3")
        tmp = self.bn(tmp, relu=True)
        tmp = self.max_pool(tmp, 3, 3, 2, 2, padding=[1, 1])

        tmp = self.proj_block(tmp, filter_num=128, padding=0, name="conv2_1")
        tmp = self.res_block(tmp, filter_num=128, padding=1, name="conv2_2")
        tmp = self.res_block(tmp, filter_num=128, padding=1, name="conv2_3")
        tmp = self.proj_block(tmp, filter_num=256, padding=1, stride=2, name="conv3_1")
        return tmp

    def res_block(self, input, filter_num, padding=0, dilation=None, name=None):  #
        tmp = self.conv(input, 1, 1, filter_num // 4, 1, 1, name=name + "_1_1_reduce")
        tmp = self.bn(tmp, relu=True)
        tmp = self.zero_padding(tmp, padding=padding)
        if dilation is None:
            tmp = self.conv(tmp, 3, 3, filter_num // 4, 1, 1, name=name + "_3_3")
        else:
            tmp = self.atrous_conv(
                tmp, 3, 3, filter_num // 4, dilation, name=name + "_3_3")
        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, 1, 1, filter_num, 1, 1, name=name + "_1_1_increase")
        tmp = self.bn(tmp, relu=False)
        tmp = input + tmp  #
        tmp = fluid.layers.relu(tmp)
        return tmp

    def proj_block(self, input, filter_num, padding=0, dilation=None, stride=1,
                   name=None):
        proj = self.conv(
            input, 1, 1, filter_num, stride, stride, name=name + "_1_1_proj")
        proj_bn = self.bn(proj, relu=False)

        tmp = self.conv(
            input, 1, 1, filter_num // 4, stride, stride, name=name + "_1_1_reduce")
        tmp = self.bn(tmp, relu=True)

        tmp = self.zero_padding(tmp, padding=padding)
        if padding == 0:
            padding = 'SAME'
        else:
            padding = 'VALID'
        if dilation is None:
            tmp = self.conv(
                tmp,
                3,
                3,
                filter_num // 4,
                1,
                1,
                padding=padding,
                name=name + "_3_3")
        else:
            tmp = self.atrous_conv(
                tmp,
                3,
                3,
                filter_num // 4,
                dilation,
                padding=padding,
                name=name + "_3_3")

        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, 1, 1, filter_num, 1, 1, name=name + "_1_1_increase")
        tmp = self.bn(tmp, relu=False)
        tmp = proj_bn + tmp
        tmp = fluid.layers.relu(tmp)
        return tmp

    def sub_net_4(self, input, input_shape):
        tmp = self.interp(input, out_shape=np.ceil(input_shape // 32))
        tmp = self.dilation_convs(tmp)
        tmp = self.pyramis_pooling(tmp, input_shape)
        tmp = self.conv(tmp, 1, 1, 256, 1, 1, name="conv5_4_k1")
        tmp = self.bn(tmp, relu=True)
        tmp = self.interp(tmp, input_shape // 16)
        return tmp

    def sub_net_2(self, input):
        tmp = self.conv(input, 1, 1, 128, 1, 1, name="conv3_1_sub2_proj")
        tmp = self.bn(tmp, relu=False)
        return tmp

    def sub_net_1(self, input):
        tmp = self.conv(input, 3, 3, 32, 2, 2, padding='SAME', name="conv1_sub1")
        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, 3, 3, 32, 2, 2, padding='SAME', name="conv2_sub1")
        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, 3, 3, 64, 2, 2, padding='SAME', name="conv3_sub1")
        tmp = self.bn(tmp, relu=True)
        tmp = self.conv(tmp, 1, 1, 128, 1, 1, name="conv3_sub1_proj")
        tmp = self.bn(tmp, relu=False)
        return tmp

    def CCF24(self, sub2_out, sub4_out, input_shape):
        tmp = self.zero_padding(sub4_out, padding=2)
        tmp = self.atrous_conv(tmp, 3, 3, 128, 2, name="conv_sub4")
        tmp = self.bn(tmp, relu=False)
        tmp = tmp + sub2_out
        tmp = fluid.layers.relu(tmp)
        tmp = self.interp(tmp, input_shape // 8)
        return tmp

    def CCF124(self, sub1_out, sub24_out, input_shape):
        tmp = self.zero_padding(sub24_out, padding=2)
        tmp = self.atrous_conv(tmp, 3, 3, 128, 2, name="conv_sub2")
        tmp = self.bn(tmp, relu=False)
        tmp = tmp + sub1_out
        tmp = fluid.layers.relu(tmp)
        tmp = self.interp(tmp, input_shape // 4)
        return tmp

    def icnet(self, data, num_classes, input_shape):
        image_sub1 = data
        image_sub2 = self.interp(data, out_shape=input_shape * 0.5)  # resize 0.5

        s_convs = self.shared_convs(image_sub2)
        sub4_out = self.sub_net_4(s_convs, input_shape)
        sub2_out = self.sub_net_2(s_convs)
        sub1_out = self.sub_net_1(image_sub1)

        sub24_out = self.CCF24(sub2_out, sub4_out, input_shape)
        sub124_out = self.CCF124(sub1_out, sub24_out, input_shape)

        conv6_cls = self.conv(
            sub124_out, 1, 1, num_classes, 1, 1, biased=True, name="conv6_cls")
        sub4_out = self.conv(
            sub4_out, 1, 1, num_classes, 1, 1, biased=True, name="sub4_out")
        sub24_out = self.conv(
            sub24_out, 1, 1, num_classes, 1, 1, biased=True, name="sub24_out")

        return sub4_out, sub24_out, conv6_cls


#######################################################
def data_create():
    def get_train_file(path):  # recursive  get data and name, return map
        for cell in os.listdir(path):
            if os.path.isdir(path + cell + "/"):
                get_train_file(path + cell + "/")
            else:
                train_data_dirs[cell] = path + cell

    def get_label_file(path):  # recursive  get data and name, return map
        for cell in os.listdir(path):
            if os.path.isdir(path + cell + "/"):
                get_label_file(path + cell + "/")
            else:
                label_dirs[cell] = path + cell

    get_train_file('./datasets/train_data/ColorImage_road02/')
    get_train_file('./datasets/train_data/ColorImage_road03/')
    get_train_file('./datasets/train_data/ColorImage_road04/')

    get_label_file('./datasets/train_label/Label_road02/')
    get_label_file('./datasets/train_label/Label_road03/')
    get_label_file('./datasets/train_label/Label_road04/')

    # get_test_file('/home/aistudio/data/data2492/TestSet.zip_files/ColorImage/')
    def init_LIST():  # get train and test file_path  detail
        for cell in train_data_dirs.keys():
            data = train_data_dirs.get(cell)
            label = label_dirs.get(cell.split('.')[0] + "_bin.png")
            TRAIN_LIST.append((data, label))

        print(TRAIN_LIST[0])

    init_LIST()
    train_reader = DataGenerater(TRAIN_LIST).create_train_reader()
    return train_reader

# train function
# this method is designed for `fit a line`
# try to update it
batch_size = 16
init_model = None

LAMBDA1 = 0.8  # lambda1
LAMBDA2 = 0.4
LAMBDA3 = 0.2
POWER = 0.9
LOG_PERIOD = 1  #
CHECKPOINT_PERIOD = 500  #
LEARNING_RATE = 0.0003
PASS_NUM = 1
len_pass = 60
TOTAL_STEP = PASS_NUM * len_pass
no_grad_set = []
def create_loss(predict, label, num_classes):
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])  # (-1, 9, 180, 360) -> (-1, 180, 360, 9)
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    label = fluid.layers.reshape(label, shape=[-1, 1])
    label = fluid.layers.cast(label, dtype="int64")
    loss = fluid.layers.softmax_with_cross_entropy(predict, label)
    no_grad_set.append(label.name)
    return fluid.layers.reduce_mean(loss)

def poly_decay():
    global_step = _decay_step_counter()
    with init_on_cpu():        # 
        decayed_lr = LEARNING_RATE * (fluid.layers.pow(
            (1 - global_step / TOTAL_STEP), POWER))
    return decayed_lr
def train(use_cuda, is_local,param_path,checkpoint_path):
    data_shape = TRAIN_DATA_SHAPE
    num_classes = NUM_CLASSES
    # define network
    images = fluid.layers.data(name='image', shape=data_shape, dtype='float32')
    label_sub1 = fluid.layers.data(name='label_sub1', shape=[1], dtype='int32')
    label_sub2 = fluid.layers.data(name='label_sub2', shape=[1], dtype='int32')
    label_sub4 = fluid.layers.data(name='label_sub4', shape=[1], dtype='int32')

    sub4_out, sub24_out, sub124_out = icnet().icnet(
        images, num_classes, np.array(data_shape[1:]).astype("float32"))
    loss_sub4 = create_loss(sub4_out, label_sub4, num_classes)
    loss_sub24 = create_loss(sub24_out, label_sub2, num_classes)
    loss_sub124 = create_loss(sub124_out, label_sub1, num_classes)
    reduced_loss = LAMBDA1 * loss_sub4 + LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124

    regularizer = fluid.regularizer.L2Decay(0.0001)
    optimizer = fluid.optimizer.SGD(learning_rate=0.001,regularization=regularizer)
    # optimizer = fluid.optimizer.Momentum(
    #     learning_rate=poly_decay(), momentum=0.9, regularization=regularizer)
    _, params_grads = optimizer.minimize(reduced_loss, no_grad_set=no_grad_set)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    if training_role == "PSERVER":
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program):
        """
        train_loop
        """
        train_reader = paddle.batch(data_create(),
                batch_size = batch_size)
        feeder = fluid.DataFeeder(place=place,feed_list=[images,label_sub1,label_sub2,label_sub4])
        exe.run(fluid.default_startup_program())

        if init_model is not None:
            print("load model from: %s" % init_model)
            sys.stdout.flush()
            fluid.io.load_params(exe, init_model)

        t_loss = 0.
        sub4_loss = 0.
        sub24_loss = 0.
        sub124_loss = 0.
        for pass_id in range(PASS_NUM):
            batch_id = 0
            for data in train_reader():  #here train_reader is function ,so
                #####?????data?????
                results = exe.run(main_program,
                                  feed=feeder.feed(data),
                    fetch_list=[reduced_loss, loss_sub4, loss_sub24, loss_sub124])
                batch_id += 1
                t_loss += results[0]
                sub4_loss += results[1]
                sub24_loss += results[2]
                sub124_loss += results[3]
                if batch_id % LOG_PERIOD == 0:
                    print(
                        "Pass[%d];Iter[%d]; train loss: %.3f; sub4_loss: %.3f; sub24_loss: %.3f; sub124_loss: %.3f"
                        % (pass_id,batch_id, t_loss / LOG_PERIOD, sub4_loss / LOG_PERIOD,
                           sub24_loss / LOG_PERIOD, sub124_loss / LOG_PERIOD))
                    t_loss = 0.
                    sub4_loss = 0.
                    sub24_loss = 0.
                    sub124_loss = 0.
                    sys.stdout.flush()

                if batch_id % CHECKPOINT_PERIOD == 0 and checkpoint_path is not None:
                    dir_name = checkpoint_path + "/" + str(pass_id) + "_" + str(batch_id)
                    fluid.io.save_persistables(exe, dirname=dir_name)
                    print("Saved checkpoint: %s" % (dir_name))
                # if batch_id == 500:
                #     model_path = param_path + "/" +str(pass_id)
                #     fluid.io.save_params(executor=exe, dirname=model_path, main_program=None)
                #     sys.exit()
            model_path = param_path + "/" +str(pass_id)
            fluid.io.save_params(executor=exe, dirname=model_path, main_program=None)
        exe.close()

    if is_local:
        train_loop(fluid.default_main_program())
    else:
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        print(pserver_endpoints)
        print(trainers)
        print(trainer_id)
        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)

        if training_role == "PSERVER":
            print("pserver")
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            print("trainer")
            train_loop(t.get_trainer_program())



# main function
def main(use_cuda, is_local=True):
    """
    main
    """
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    param_path = "./output/model/"
    checkpoint_path = "./datasets/checkpoint/"
    if not os.path.isdir(param_path):
        os.makedirs(param_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    train(use_cuda, is_local,param_path,checkpoint_path)


if __name__ == '__main__':
     main(True, False)
