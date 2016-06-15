import caffe
from caffe import layers as L
from caffe import params as P
import tempfile

EltwiseParameter_EltwiseOp_PROD = 0
EltwiseParameter_EltwiseOp_SUM = 1
EltwiseParameter_EltwiseOp_MAX = 2


def wavelet_layer(angles=[0], output_path=None):
    n = caffe.NetSpec()

    n.data = L.DummyData(shape=dict(dim=[3, 1, 256, 256]))

    n.conv1 = L.Convolution(n.data, kernel_size=16, stride=1,
                            num_output=1,
                            weight_filler=dict(type='constant', value=0).copy(),
                            bias_filler=dict(type='constant', value=0).copy())

    n.power1 = L.Power(n.conv1, power=2)

    n.conv2 = L.Convolution(n.data, kernel_size=16, stride=1,
                            num_output=1,
                            weight_filler=dict(type='constant', value=0).copy(),
                            bias_filler=dict(type='constant', value=0).copy())
    n.power2 = L.Power(n.conv2, power=2)

    n.add = L.Eltwise(n.power1,
                      n.power2,
                      eltwise_param=dict(operation=EltwiseParameter_EltwiseOp_SUM))

    n.sqrt = L.Power(n.add, power=.5)

    if output_path:
        with open(output_path, 'w+') as f:
            f.write(str(n.to_proto()))
            return f.name
    else:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(str(n.to_proto()))
            return f.name

