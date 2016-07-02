import numpy as np

import caffe
from caffe import layers as L
from caffe import params as P
import tempfile
import operator
import time
import os
import wavelet

EltwiseParameter_EltwiseOp_PROD = 0
EltwiseParameter_EltwiseOp_SUM = 1
EltwiseParameter_EltwiseOp_MAX = 2


def conv_layer(bottom, dim, kernel_size, name, group=1, stride=1, pad=0):
    return L.Convolution(
        bottom,
        name=name,
        group=group,
        convolution_param=dict(
            num_output=dim, kernel_size=kernel_size,
            stride=stride,
            pad=pad,
            weight_filler=dict(type="constant", value=0),
            bias_filler=dict(type="constant", value=0)))


def add(in1, in2):
    return L.Eltwise(
        in1, in2,
        eltwise_param=dict(operation=1))


def scat_layer(bottom, dim, kernel_size, name, group=1):
    conv1 = conv_layer(bottom, dim, kernel_size, name + '_real', group=group)
    pow1 = L.Power(conv1, power=2, in_place=True)
    conv2 = conv_layer(bottom, dim, kernel_size, name + '_imag', group=group)
    pow2 = L.Power(conv2, power=2, in_place=True)
    res_add = add(pow1, pow2)
    res_add = L.Power(res_add, power=.5, in_place=True)
    return res_add


def gen_prototxt(nangles,
                 max_order,
                 scales,
                 filter_size_factor=wavelet.DEFAULT_SIZE,
                 nchannels_input=3,
                 data=None,
                 verbose=False,
                 output_path=None):
    n = caffe.NetSpec()
    if data is None:
        data = L.Input(shape=dict(dim=[1, nchannels_input, 256, 256]))

    n.data = data

    scat_count = -1
    dim_total = nchannels_input
    layers = [[(data, [None], 0)]]
    for o in range(max_order):
        layer = []
        for s in scales:
            kernel_size = s * filter_size_factor * 2
            delta_offset = kernel_size // 2

            for c0, s0, offset in layers[-1]:
                if s0[-1] is not None and s <= s0[-1]:
                    continue
                scat_count += 1
                dim_in = nchannels_input * nangles ** o
                dim_out = nchannels_input * nangles ** (o + 1)
                dim_total += dim_out
                name = 'scat%i_%i_%ito%i' % (s, scat_count, dim_in, dim_out)

                c = scat_layer(c0,
                               dim=dim_out,
                               kernel_size=kernel_size,
                               name=name,
                               group=dim_in)

                layer.append((c, s0 + [s], offset + delta_offset))

                if verbose:
                    print "%s:" % name
                    print "  kernel size: %i" % kernel_size
                    print "  %s (%i)" % ("->".join(map(str, (s0 + [s]))), dim_out)

        layers.append(layer)

    if verbose:
        print "Total output dimensionality: %i" % dim_total

    # Crop the coefficients before concatenation
    # The last coefficient is the smallest because it's having the highest order.
    last_coefficient = layers[-1][-1][0]
    max_offset = layers[-1][-1][2]
    coefficients = []
    for layer in layers:
        for c, _, offset in layer:
            coefficients.append(L.Crop(c, last_coefficient, offset=max_offset - offset))

    concat = L.Concat(*coefficients)

    # Do the final gaussian blur and resampling
    kernel_size = scales[-1] * filter_size_factor * 2
    stride = scales[-1]
    c = conv_layer(concat,
                   dim=dim_total,
                   group=dim_total,
                   kernel_size=kernel_size,
                   name='psi',
                   stride=stride)

    n.output = c

    proto_str = str(n.to_proto())

    if output_path:
        with open(output_path, 'w+') as f:
            f.write(proto_str)
            return f.name
    else:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(proto_str)
            return f.name


def generate_filters(net, **kwargs):
    nangles = kwargs['nangles']
    scales = kwargs['scales']
    filter_size_factor = kwargs.get('filter_size_factor', wavelet.DEFAULT_SIZE)

    for i, s in enumerate(scales):
        name = "scat%1i" % s
        keys = [k for k in net.params.keys() if name in k]

        for ai, a in enumerate(np.linspace(0, np.pi, nangles, endpoint=False)):
            kernel_size = s * filter_size_factor * 2
            kernel = wavelet.morlet(s, a, kernel_size)

            for k in keys:
                if '_real' in k:
                    net.params[k][0].data[ai::nangles, :, :, :] = np.real(kernel)
                elif '_imag' in k:
                    net.params[k][0].data[ai::nangles, :, :, :] = np.imag(kernel)

    s = scales[-1]
    kernel_size = s * filter_size_factor * 2
    gauss_kernel = wavelet.gauss_kernel(s, kernel_size)
    net.params['psi'][0].data[:, :, :, :] = gauss_kernel


def scatnet(**kwargs):
    net = caffe.Net(gen_prototxt(**kwargs), caffe.TEST)
    generate_filters(net, **kwargs)
    return net


def get_arg_string(**kwargs):
    args = []
    args.append("a%s" % str(kwargs['nangles']))
    args.append("m%s" % str(kwargs['max_order']))
    args.append("s%s" % "s".join(map(str, kwargs['scales'])))
    args.append("f%s" % str(kwargs['filter_size_factor']))
    return "_".join(args)


def get_layers_sizes(nangles,
                     max_order,
                     scales,
                     nchannels_input=3):
    dim_total = nchannels_input
    layers = [[(dim_total, 0)]]
    for o in range(max_order):
        layer = []
        for s in scales:
            for c0, s0 in layers[-1]:
                if s0 is not None and s <= s0:
                    continue
                dim_out = nchannels_input * nangles ** (o + 1)
                dim_total += dim_out

                layer.append((dim_out, s))

        layers.append(layer)
    steps = [x / 3 for x in zip(*reduce(operator.add, layers))[0]]

    mask = np.zeros(dim_total, dtype='int')
    i = 0
    for d in steps:
        for c in range(nchannels_input):
            mask[i:i + d] = c
            i += d

    return mask


def process_multiple(*args, **kwargs):
    # TODO: Forward propagate images
    # TODO: Normalize and convert to uint8
    # TODO: Write to LMDB
    # TODO: produce mean value
    # TODO: Save model and prototxt


    caffe.set_mode_gpu()

    input_path = kwargs.pop('input_path')

    output_dir_path = os.path.join(kwargs.pop('output_path'), get_arg_string(**kwargs))
    output_float_path = os.path.join(output_dir_path, 'float32')
    output_info_list = os.path.join(output_dir_path, 'list.txt')
    output_lmdb_path = os.path.join(output_dir_path, 'lmdb')
    output_mean_path = os.path.join(output_dir_path, 'mean_coefficient.npy')
    output_mag_path = os.path.join(output_dir_path, 'coefficient_magnitude.npy')

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    if not os.path.exists(output_float_path):
        os.makedirs(output_float_path)

    nimages = kwargs.pop('nimages', None)
    image_infos = open(input_path, 'r').read().split('\n')
    if nimages is None:
        nimages = len(image_infos)
    else:
        image_infos = image_infos[:nimages]

    # Prepare network
    transform_param = dict(mirror=False, crop_size=227)
    species_data, species_label = L.ImageData(transform_param=transform_param,
                                              source=input_path,
                                              batch_size=1,
                                              new_height=256,
                                              new_width=256,
                                              ntop=2)
    net = scatnet(data=species_data, **kwargs)
    shape = net.blobs['output'].data.shape[1:]

    print "Output folder:\n %s" % output_dir_path
    print kwargs

    mean_coefficient = np.zeros(shape=shape, dtype='float32')
    coeff_magnitudes = np.zeros((shape[0], nimages), dtype='float32')

    info_file = open(output_info_list, 'w+')

    print "Generate Scattering Coefficients"
    t0 = time.time()
    for i, image_info in enumerate(image_infos):
        if not i % 100:
            dt = time.time() - t0
            progress_str = "%i/%i" % (i, nimages)
            fps = i / dt
            proc_time = 1000 * dt / i if i > 0 else 0
            msg = "%11s, %5.1f img/s, %5.1f ms/img" % (progress_str, fps, proc_time)
            print msg

        net.forward()
        output = net.blobs['output'].data[0,...].copy()

        mean_coefficient += output
        coeff_magnitudes[:,i] = np.sqrt(np.sum(np.sum(output**2,1),1))

        base = os.path.basename(image_info.split()[0])
        label = image_info.split()[1]

        file_name = os.path.splitext(base)[0] + '.npy'
        file_path = os.path.join(output_float_path, file_name)
        output.tofile(file_path)

        info_file.write("%s %s\n" % (file_path, label))

    open(os.path.join(output_dir_path, 'shape.txt'),'w+').write(str(shape))

    print "Save mean coefficients and mean magnitude"
    mean_coefficient /= nimages
    mean_coefficient.tofile(output_mean_path)
    coeff_magnitudes.tofile(output_mag_path)


if __name__ == '__main__':
    # TODO: Support an explicit list of scales
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('-o', '--output_path', required=True, type=str)
    parser.add_argument('-a', '--nangles', type=int, default=4)
    parser.add_argument('-s', '--scale', type=int, default=3)
    parser.add_argument('-m', '--max_order', type=int, default=3)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    parser.add_argument('--nimages', type=int, default=None)
    parser.add_argument('--filter_size_factor', type=int, default=2)
    args = parser.parse_args()

    scales = 2 ** np.arange(0, args.scale)

    process_multiple(input_path=args.input_path,
                     output_path=args.output_path,
                     scales=scales,
                     max_order=args.max_order,
                     nangles=args.nangles,
                     verbose=args.verbose,
                     nimages=args.nimages,
                     filter_size_factor=args.filter_size_factor)
