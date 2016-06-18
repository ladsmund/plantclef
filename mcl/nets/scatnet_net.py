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


def conv_layer(bottom, dim, kernel_size, name, group=1, stride=1, pad=None):
    if pad is None:
        # pad = kernel_size // 2
        pad = 0
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


def gen_prototxt(nangles, max_order, scales, kernel_size, data=L.Input(shape=dict(dim=[1, 1, 256, 256])),
                 verbose=False,
                 output_path=None):
    n = caffe.NetSpec()
    n.data = data

    nscales = len(scales)
    delta_offset = kernel_size // 2

    global scat_count
    scat_count = -1

    def new_scat_layer(input, id, dim_out, dim_in):
        global scat_count
        scat_count += 1
        return scat_layer(input, dim=dim_out, group=dim_in, kernel_size=kernel_size,
                          name='scat%i_%i_%ito%i' % (id, scat_count, dim_in, dim_out))

    dim_total = 1
    layers = [[(data, [None],0)]]
    for o in range(max_order):
        layer = []
        for s in range(nscales):
            for c0, s0, offset in layers[-1]:
                if s0[-1] is not None and s <= s0[-1]:
                    continue

                dim_in = nangles ** o
                dim_out = nangles ** (o + 1)
                dim_total += dim_out
                c = new_scat_layer(c0, s, dim_out, dim_in)
                layer.append((c, s0 + [s], offset+delta_offset))

                if verbose:
                    print "%s (%i)" % ("->".join(map(str, (s0 + [s]))), dim_out)

        layers.append(layer)


    last_coefficient = layers[-1][-1][0]
    max_offset = layers[-1][-1][2]

    coefficients = []
    for layer in layers:
        for c, _, offset in layer:
            print c
            coefficients.append(L.Crop(c, last_coefficient, offset=max_offset-offset))

    # layers = zip(*reduce(operator.add, layers))[0]
    # layers = [L.Crop(l, layers[-1]) for l in layers]

    concat = L.Concat(*coefficients)

    print "Total output dimensionality: %i" % dim_total

    stride = 2 ** (max(scales))
    # dilation

    c = conv_layer(concat,
                   dim=dim_total,
                   group=dim_total,
                   kernel_size=kernel_size,
                   name='psi',
                   stride=stride)

    n.output = L.Crop(c, c, offset=0)
    # n.output = L.Crop(c, crop_param=dict(offset=10))

    # n.slices = L.Slice(n.output, slice_point=[10,20,30], ntop=1, axis=2)
    # slices = L.Slice(n.output, slice_point=[10,20,30], ntop=1, axis=1)
    # for i in range(4):
    #     setattr(n, "test_%i"%i, slices[i])

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
    kernel_size = kwargs['kernel_size']

    for i, s in enumerate(scales):
        name = "scat%1i" % i

        keys = [k for k in net.params.keys() if name in k]

        for ai, a in enumerate(np.linspace(0, np.pi, nangles, endpoint=False)):
            kernel = wavelet.morlet(2 ** (s), a, kernel_size + 1)
            kernel = kernel[1:, 1:]
            for k in keys:
                if '_real' in k:
                    net.params[k][0].data[ai::nangles, :, :, :] = np.real(kernel)
                elif '_imag' in k:
                    net.params[k][0].data[ai::nangles, :, :, :] = np.imag(kernel)

    gauss_kernel = wavelet.gauss_kernel(2 ** scales[-1], kernel_size + 1)[1:, 1:]
    net.params['psi'][0].data[:, :, :, :] = gauss_kernel


def scatnet(**kwargs):
    net = caffe.Net(gen_prototxt(**kwargs), caffe.TEST)
    generate_filters(net, **kwargs)
    return net


if __name__ == '__main__':

    scale = 3
    scales = range(scale)
    max_order = 3
    nangles = 6
    kernel_size = 15

    net = scatnet(scales=scales, max_order=max_order, nangles=nangles, kernel_size=kernel_size, verbose=True)
    print "\n\n" + "-" * 30
    for k in net.blobs.keys():
        if "split" in k:
            print "%s: %s" % (k, str(net.blobs[k].data.shape))
