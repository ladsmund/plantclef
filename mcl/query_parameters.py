import caffe

exps = ['20160624-121155-2c79','20160624-182808-811a','20160625-001322-6968','20160625-083048-3086',
'20160703-150520-3997','20160703-164719-37a7','20160703-165031-26a6','20160703-184038-b35f',
'20160703-195208-cdcf','20160703-195348-6004','20160703-211919-c6bb','20160703-232402-8605',
'20160704-001413-898d','20160704-001627-a4de','20160704-001701-30e0','20160704-001743-6578',
'20160704-001911-eb45','20160704-065721-1c42','20160704-090616-ca2e','20160704-104545-8c3e',
'20160704-144437-f9a3','20160704-152854-7d95','20160704-154710-e898','20160704-162710-6dc9',
'20160704-171110-ef79','20160704-171221-2222','20160704-232348-6577','20160704-232921-df78',
'20160705-071507-f290','20160705-072003-d85d','20160705-075154-d89f','20160705-091349-2744',
'20160705-130445-443c','20160705-131412-ff07','20160705-214703-ac3e','20160705-214721-4d47',
'20160705-214733-e267','20160705-214758-e0a4','20160705-214810-44a7','20160705-214821-0e83',
'20160705-214944-ddc0','20160705-215037-3a1c','20160705-220738-8a08','20160705-221737-ca2c',
'20160705-221756-e7b1','20160705-221918-6a5b','20160706-065616-d8f8','20160708-003123-0d2e',
'20160708-003847-e5ae','20160708-110212-1143']

def query_parameters(input_path):
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
    parser.add_argument('input_paths', nargs='+')
    args = parser.parse_args()

    for input_path in args.input_paths:
        proto_path = os.path.join(input_path, 'train_val.prototxt')
        base = os.path.basename(input_path)

        if base not in exps:
            continue

        try:
            res = str(query_parameters(proto_path))
        except RuntimeError as e:
            res = 'Skipped'

        print base, res

