import scipy
import numpy as np
import caffe
import pickle


def vis_square(data):
    """
    Copied from Caffe tutorial 00-classification
    Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return data


def query_parameters(net, caffemodel=None):
    if caffemodel:
        net.copy_from(caffemodel)

    params = dict()
    for k in net.params.keys():
        params[k] = net.params[k][0].data.copy()
    return params


def query_parameters_old(proto_path):
    net = caffe.Net(proto_path, caffe.TRAIN)

    params = dict()
    for k in net.params.keys():
        params[k] = []
        for i, p in enumerate(net.params[k]):
            params[k].append(p.data.shape)
    return params


def save_filter_image(image_path, filter):
    try:
        filter_image = vis_square(filter.transpose(0, 2, 3, 1)[:, :, :, ::-1])
        print "save %s" % image_path
        scipy.misc.imsave(image_path, filter_image)
    except Exception as e:
        print "Error: ", e.message()


if __name__ == '__main__':
    import os
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser('Analyze filters')
    parser.add_argument('input_path')
    parser.add_argument('base_model_path')
    parser.add_argument('output_path')
    parser.add_argument('-e', '--exclude_path')
    parser.add_argument('--export_filter')
    parser.add_argument('-r', '--redo', type=bool, default=False)

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    proto_path = os.path.join(input_path, 'train_val.prototxt')
    caffemodels_list = glob(os.path.join(input_path, '*.caffemodel'))
    caffemodels = {int(filename.split('.')[0].split('_')[-1]): filename for filename in caffemodels_list}

    net_id = os.path.basename(input_path)

    basemodel = args.base_model_path
    if basemodel == 'None':
        basemodel = None
    export_filter = args.export_filter

    net = caffe.Net(proto_path, caffe.TRAIN)
    
    param_base = None 
    if basemodel:
        param_base = query_parameters(net, basemodel)

        if export_filter in param_base:
            image_name = "%s_0.png" % net_id
            image_path = os.path.join(output_path, image_name)
            save_filter_image(image_path, param_base[export_filter])

    all_diffs = dict()
    print "**" * 30
    print "loading models"
    for iteration, m in caffemodels.items():
        param = query_parameters(net, caffemodel=m)

        if export_filter in param:
            image_name = "%s_%i.png" % (net_id, iteration)
            image_path = os.path.join(output_path, image_name)
            save_filter_image(image_path, param[export_filter])

        if param_base is None:
            param_base = param
        param_diff = dict()
        for k in param.keys():
            param_diff[k] = np.sum((param[k] - param_base[k]) ** 2)

        all_diffs[iteration] = param_diff

        print "-" * 30
        print iteration

    pickle.dump(all_diffs, open(os.path.join(output_path, "filter_diff_%s.pickle" % net_id),'w+'))

    for iteration, v in sorted(all_diffs.items()):
        print iteration
        for k, f_diff in sorted(v.items()):
            print "  %s: %4.f" % (k, f_diff)
