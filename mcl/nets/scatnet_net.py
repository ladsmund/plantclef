import numpy as np

import caffe
from caffe import layers as L
from caffe import params as P
import tempfile
import operator

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
            kernel_size = s * filter_size_factor
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
    kernel_size = scales[-1] * filter_size_factor
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
            kernel_size = s * filter_size_factor
            kernel = wavelet.morlet(s, a, kernel_size)

            for k in keys:
                if '_real' in k:
                    net.params[k][0].data[ai::nangles, :, :, :] = np.real(kernel)
                elif '_imag' in k:
                    net.params[k][0].data[ai::nangles, :, :, :] = np.imag(kernel)

    s = scales[-1]
    kernel_size = s * filter_size_factor
    gauss_kernel = wavelet.gauss_kernel(s, kernel_size)
    net.params['psi'][0].data[:, :, :, :] = gauss_kernel


def scatnet(**kwargs):
    net = caffe.Net(gen_prototxt(**kwargs), caffe.TEST)
    generate_filters(net, **kwargs)
    return net


def split_to_input_channels(input, nangles,
                 max_order,
                 scales,
                 nchannels_input=3):


    dim_total = nchannels_input
    layers = [[dim_total, 0]]
    for o in range(max_order):
        layer = []
        for s in scales:

            for c0, s0, offset in layers[-1]:
                if s0[-1] is not None and s <= s0[-1]:
                    continue
                dim_out = nchannels_input * nangles ** (o + 1)
                dim_total += dim_out

                layer.append((dim_out, s0 + [s]))

        layers.append(layer)


if __name__ == '__main__':

    scale = 3
    scales = 2 ** np.arange(0, scale)
    max_order = 3
    nangles = 6
    filter_size_factor = 3

    net = scatnet(scales=scales, max_order=max_order, nangles=nangles, verbose=True,
                  filter_size_factor=filter_size_factor)
    print "\n\n" + "-" * 30
    for k in net.blobs.keys():
        if "split" in k:
            print "%s: %s" % (k, str(net.blobs[k].data.shape))
