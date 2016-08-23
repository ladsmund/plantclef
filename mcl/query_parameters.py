import caffe

def query_parameters(input_path):
    caffe.set_mode_cpu()
    net = caffe.Net(input_path, caffe.TRAIN)

    params = dict()
    for k in net.params.keys():
        params[k] = []
        for i, p in enumerate(net.params[k]):
            params[k].append(p.data.shape)
    return params

if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('-e','--exclude_path')
    parser.add_argument('-r','--redo', type=bool, default=False)

    args = parser.parse_args()


    if os.path.exists(args.exclude_path) and not args.redo:
        lines = open(args.exclude_path,'r').readlines()
        results = {l.split()[0]: l.split()[1:] for l in lines}
    else:
        results = dict()

    input_path = args.input_path
    proto_path = os.path.join(input_path, 'train_val.prototxt')
    base = os.path.basename(input_path)

    # if base not in exps:
    #     continue

    if base in results.keys():
        exit()

    try:
        res = str(query_parameters(proto_path))
    except RuntimeError as e:
        res = 'Skipped'

    print base, res
