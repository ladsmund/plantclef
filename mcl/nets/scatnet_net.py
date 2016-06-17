import caffe
from caffe import layers as L
from caffe import params as P
import tempfile
import operator

EltwiseParameter_EltwiseOp_PROD = 0
EltwiseParameter_EltwiseOp_SUM = 1
EltwiseParameter_EltwiseOp_MAX = 2


def conv_layer(bottom, dim, kernel_size, name, group=1, stride=1, pad=None):
    if pad is None:
        pad = kernel_size // 2
    return L.Convolution(
        bottom,
        name=name,
        group=group,
        convolution_param=dict(
            num_output=dim, kernel_size=kernel_size,
            stride=stride, pad=pad,
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
                 output_path=None):
    n = caffe.NetSpec()
    n.data = data

    nscales = len(scales)

    global scat_count
    scat_count = -1

    def new_scat_layer(input, id, dim_out, dim_in):
        global scat_count
        scat_count += 1
        return scat_layer(input, dim=dim_out, group=dim_in, kernel_size=kernel_size,
                          name='scat%i_%i_%ito%i' % (id, scat_count, dim_in, dim_out))

    dim_total = 1
    layers = [[(data, [None])]]
    for o in range(max_order):
        layer = []
        for s in range(nscales):
            for c0, s0 in layers[-1]:
                if s0[-1] is not None and s <= s0[-1]:
                    continue

                dim_in = nangles ** o
                dim_out = nangles ** (o + 1)
                dim_total += dim_out
                c = new_scat_layer(c0, s, dim_out, dim_in)
                layer.append((c, s0 + [s]))

                print "%s (%i)" % ("->".join(map(str, (s0 + [s]))), dim_out)

        layers.append(layer)

    layers = zip(*reduce(operator.add, layers))[0]
    concat = L.Concat(*layers)

    print "Total output dimensionality: %i" % dim_total

    stride = 2 ** (max(scales))
    n.output = conv_layer(concat,
                          dim=dim_total,
                          group=dim_total,
                          kernel_size=kernel_size,
                          name='psi',
                          stride=stride)

    # slices = L.Slice(n.output, slice_point=[10,20,30], ntop=4)
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


def scatnet(*args, **kwargs):
    return caffe.Net(gen_prototxt(*args, **kwargs), caffe.TEST)
