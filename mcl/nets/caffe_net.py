import caffe
from caffe import layers as L
from caffe import params as P

import tempfile

DEFAULT_BATCH_SIZE = 32


transform_param = dict(mirror=False, crop_size=227)

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01).copy(),
              bias_filler=dict(type='constant', value=0.1).copy()):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005).copy(),
            bias_filler=dict(type='constant', value=0.1).copy()):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False, output_path=None, return_string=False,
             learning_level=0):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    proto_string = str(n.to_proto())
    if return_string:
        return proto_string
    if output_path:
        with open(output_path, 'w+') as f:
            f.write(proto_string)
            return f.name
    else:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(proto_string)
            return f.name


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('-t', '--data_type', type=str, default='lmdb')
    parser.add_argument('-l', '--learning_level', default=0)
    parser.add_argument('--learn_all', default=False, action='store_true')
    parser.add_argument('-m', '--mirror', type=bool, default=False)
    parser.add_argument('-c', '--nclasses', type=int, default=1000)
    parser.add_argument('-n', '--name', type=str, default='fc8')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=True)
    parser.add_argument('-b', '--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    transform_param = dict(mirror=args.mirror, crop_size=227, mean_value=128, scale=1./128)

    if args.data_type == 'list':
        data_layer, label_layer = L.ImageData(
            transform_param=transform_param,
            source=args.source,
            batch_size=args.batch_size,
            new_height=256,
            new_width=256,
            ntop=2)

    elif args.data_type == 'lmdb':
        data_layer, label_layer = L.Data(
            transform_param=transform_param,
            batch_size=args.batch_size,
            backend=P.Data.LMDB,
            source=args.source,
            ntop=2)
    else:
        raise Exception('unsupported data type %s' % args.data_type)

    proto_string = caffenet(data=data_layer,
                            label=label_layer,
                            train=args.train,
                            num_classes=args.nclasses,
                            learn_all=args.learn_all,
                            learning_level=args.learning_level,
                            classifier_name=args.name,
                            return_string=True)

    sys.stdout.write(proto_string)
