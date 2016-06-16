import caffe
from caffe import layers as L
from caffe import params as P
import tempfile
import operator

EltwiseParameter_EltwiseOp_PROD = 0
EltwiseParameter_EltwiseOp_SUM = 1
EltwiseParameter_EltwiseOp_MAX = 2


def conv_layer(bottom, dim, kernel_size, name, group=1, stride=1):
    return L.Convolution(
        bottom,
        name=name,
        group=group,
        convolution_param=dict(
            num_output=dim, kernel_size=kernel_size,
            stride=stride, pad=kernel_size // 2,
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


def gen_prototxt(nangles=1, max_order=1, scales=[1], data=L.Input(shape=dict(dim=[1, 1, 100, 100])),
                 output_path=None):
    n = caffe.NetSpec()
    n.data = data

    nscales = len(scales)

    def new_scat_layer(input, id, dim_out, dim_in):
        return scat_layer(input, dim=dim_out, group=dim_in, kernel_size=15, name='scat%i' % id)

    dim_total = 1
    layers = [[(data, -1)]]
    for o in range(max_order):
        layer = []
        for s in range(nscales):
            for c0, s0 in layers[-1]:
                if s <= s0:
                    continue
                dim_in = nangles ** o
                dim_out = nangles ** (o + 1)
                dim_total += dim_out
                c = new_scat_layer(c0, s, dim_out, dim_in)
                layer.append((c, s))
        layers.append(layer)

    layers = zip(*reduce(operator.add, layers))[0]
    concat = L.Concat(*layers)

    print "Total output dimensionality: %i" % dim_total



    # o1s1 = scat_layer(n.data, dim=6, kernel_size=15, name='scat0')
    # o1s2 = scat_layer(n.data, dim=6, kernel_size=15, name='scat1')
    # o1s3 = scat_layer(n.data, dim=6, kernel_size=15, name='scat2')
    #
    # o2s2 = scat_layer(o1s1, dim=36, kernel_size=15, name='scat1', group=6)
    # o2s3 = scat_layer(o1s2, dim=36, kernel_size=15, name='scat2', group=6)
    #
    # o3s3 = scat_layer(o2s2, dim=216, kernel_size=15, name='scat2', group=36)

    # n.ouet = o3s3

    # concat = L.Concat(n.data, o1s1, o1s2, o1s3, o2s2, o2s3, o3s3)

    # n.output = concat

    stride = 2**(max(scales)-1)
    print "stride: %i" % stride
    n.output = conv_layer(concat, dim=dim_total, group=dim_total, kernel_size=15, name='psi', stride=stride)
    # n.output = conv_layer(concat, dim=306, group=1, kernel_size=15, name='psi', stride=8)
    # n.output = L.Pooling(concat, kernel_size=7, stride=4, pool=P.Pooling.MAX)
    # n.output = conv_layer(concat, dim=1, kernel_size=7, name='psi', stride=8)

    # proto_str = str(n.to_proto())
    # proto_path = "./test.prototxt"
    # with open(proto_path, 'w+') as f:
    #     f.write(proto_str)
    #
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
