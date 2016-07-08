from caffe.proto import caffe_pb2
import tempfile

def tutorial(train_net_path,
             test_net_path=None,
             base_lr=0.001,
             snapshot_prefix='caffe_snapshot', return_string=False):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100)  # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1

    s.max_iter = 100000  # # of times to update the net (training iterations)

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 2000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 100

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = snapshot_prefix

    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    proto_string = str(s)
    if return_string:
        return proto_string

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(proto_string)
        return f.name



if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('train_net')
    parser.add_argument('--test_net', default=None)
    parser.add_argument('--base_lr', default=0.001)
    parser.add_argument('--snapshot_prefix', default='caffe_snapshot')
    args = parser.parse_args()
    proto_string = tutorial(train_net_path=args.train_net,
                            test_net_path=args.test_net,
                            base_lr=args.base_lr,
                            snapshot_prefix=args.snapshot_prefix,
                            return_string=True)
    sys.stdout.write(proto_string)