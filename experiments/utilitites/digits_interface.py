import os
import urllib
import json
import ast
import operator
import re
import sqlite3
from prototxt_parser import parse

experiments_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
db_path = os.path.join(experiments_folder, 'experiments.db')
data_folder = os.path.join(experiments_folder, 'data')
models_path = os.path.join(data_folder, 'digits_models.py')
digits_servers = {0: 'http://localhost:5000', 1: 'http://localhost:5002'}
key_map = {
    'id': 'net_id',
    'name': 'name',
    'node': 'node',
    'batch_size': 'batch_size',
    'solver_type': 'solver_type',
    'solver_proto': 'solver_proto',
    'net_proto': 'net_proto',
    'accuracy (val) max': 'accuracy_max',
    'accuracy (val) last': 'accuracy_val_last',
    'learning_rate (train) max': 'learning_rate_train_max',
    'progress': 'progress',
    'epoch (val) last': 'epoch_last',
    'status': 'status',
    'loss (train) last': 'loss_train_last',
    'loss (val) last': 'loss_val_last',
    'loss (val) min': 'loss_val_min',
    'learning_rate (train) last': 'learning_rate_train_last',
    'learning_rate (train) min': 'learning_rate_train_min',
    'loss (train) min': 'loss_train_min',
    'caffe_log': 'caffe_log',}
parameter_info_files = ['parameter_info_ffwgpu1.txt', 'parameter_info_imgpu.txt']


def _convert_model_keys(m_raw):
    return {key_map[k]: v for k, v in m_raw.items() if k in key_map}


def _save_models(models, path=models_path):
    f = open(path, 'w+')
    f.write('[')
    for l in map(str, models):
        f.write(l)
        f.write(',\n')
    f.write(']')


def fetch_digit_models(servers=digits_servers.copy()):
    models = []
    for i, addr in servers.items():
        print i, addr
        response = urllib.urlopen(addr + "/completed_jobs.json")
        data = json.loads(response.read())


        for m in data['models']:
            m['node'] = i

            if 'mnist' in m['name']:
                continue

            models.append(_convert_model_keys(m))

    models = filter(lambda m: m['status'] != 'Error', models)
    models = filter(lambda m: m['name'] not in ['LeNet'], models)

    return models


def load_digits_model(path=models_path):
    return ast.literal_eval(open(path).read())


def parse_prototxt(prototxt_str):
    layers = re.findall('yer {(.*?)}.la', prototxt_str, flags=re.DOTALL)
    layer_dicts = []

    layer_number = 0
    for l in layers:
        layer_dict = {}

        layer_dict['layer_number'] = layer_number
        layer_number += 1

        name = re.findall('name: "(.*?)"', l)[0]
        layer_dict['layer_name'] = name
        layer_type = re.findall('type: "(.*?)"', l)[0]
        layer_dict['type'] = layer_type

        phase = re.findall('phase: (.*?)\n', l)
        if phase:
            layer_dict['phase'] = phase[0]

        batch_size = re.findall('batch_size: (.*?)\n', l)
        if batch_size:
            layer_dict['batch_size'] = batch_size[0]

        lr_mult = re.findall('lr_mult: (.*?)\n', l)
        if lr_mult:
            layer_dict['lr_mult'] = lr_mult

        decay_mult = re.findall('decay_mult: (.*?)\n', l)
        if decay_mult:
            layer_dict['decay_mult'] = decay_mult

        layer_dicts.append(layer_dict)

    return layer_dicts


def get_param_info(net_id):
    for name in parameter_info_files:
        path = os.path.join(data_folder, name)
        for l in open(path).readlines():
            key = l.split(' ', 1)[0]
            if key == net_id and not "Skipped" in l.split(' ', 1)[1]:
                try:
                    return ast.literal_eval(l.split(' ', 1)[1])
                except ValueError as e:
                    print "Value Error:"
                    print l.split(' ', 1)[1]
                    raise e

    return None


def format_parameter(params):
    if params is None:
        return None
    return ",".join(map(lambda s: s[1:-1] if s[0] == "\"" else s, params))


def get_log_info(net_id):
    log_path = os.path.join(data_folder, '%s_log.log' % net_id)
    log_str = open(log_path).read()

    batch_size = None
    layer_number = -1
    name = None
    shapes = []
    layer_shapes = {}
    active = False
    for l in log_str.split('\n'):
        if 'Creating training net' in l:
            active = True
        elif 'Creating test net' in l:
            active = False

        if not active:
            continue

        name_iter = re.search('net.cpp:153] Setting up (.*)', l)
        if name_iter:
            new_name = name_iter.groups(0)[0]

            if name is not None:
                layer_shapes[name] = shapes

            name = new_name
            layer_number += 1
            shapes = []

        shape_iter = re.search('Top shape: ([0-9 ]+)', l)
        if shape_iter:
            shape = tuple(map(int,shape_iter.groups(0)[0].split()))
            if 'data' in name:
                batch_size = int(shape[0])
            shapes.append(shape[1:])

    return layer_shapes, batch_size


def get_net_info(net_id):
    net_path = os.path.join(data_folder, '%s_net.prototxt' % net_id)

    top_shapes, batch_size = get_log_info(net_id)

    batch_size_max = 0

    layer_params = get_param_info(net_id)
    prototxt = open(net_path).read()
    layers = []

    for layer_number, l in enumerate(parse(iter(prototxt.splitlines()))['layer']):
        layer = dict()

        # if 'data_param' in l:
        #     batch_size = int(l['data_param'][0].get('batch_size', [0])[0])
        #     batch_size_max = max(batch_size, batch_size_max)

        layer['net_id'] = net_id
        layer['layer_number'] = layer_number
        layer_name = format_parameter(l.get('name'))
        layer['layer_name'] = layer_name

        if layer_name in top_shapes:
            layer['top_shape'] = top_shapes[layer_name]

        layer['top'] = format_parameter(l.get('top'))
        layer['bottom'] = format_parameter(l.get('bottom'))

        if 'convolution_param' in l:
            conv_params = l['convolution_param'][0]
            layer['kernel_size'] = conv_params.get('kernel_size')
            layer['stride'] = conv_params.get('stride')

            if 'group' in conv_params:
                layer['groups'] = int(conv_params['group'][0])
            else:
                layer['groups'] = None

        nparams = 0
        if layer_params:
            layer['params'] = layer_params.get(layer_name, [])
            nparams = 0
            for parameter_shapes in layer['params']:
                nparams += reduce(operator.mul, parameter_shapes)
            layer['nparams'] = nparams

        max_lr = None
        for p in l.get('param', []):
            lr = float(p.get('lr_mult', [0])[0])
            max_lr = lr if max_lr is None else max(max_lr, lr)

        if max_lr is None:
            layer['lock'] = None
        elif max_lr == 0:
            layer['lock'] = 1
        else:
            layer['lock'] = 0

        layer['solver_config'] = str(l.get('param', []))

        layers.append(layer)

        if nparams == 0:
            layer['trans'] = 0
        else:
            layer['trans'] = 0 if layer_name.split('_')[-1] == 'clean' else 1

    return dict(layers=layers, batch_size=batch_size)


def save_as_sqlite(models, db_path=db_path):
    db = sqlite3.Connection(db_path)

    for m in models:
        for l in m['layers']:
            attributes = []
            values = []
            for k, v in l.items():
                if v is None:
                    continue
                attributes.append(k)
                values.append(v)

            attributes_str = ", ".join(map(str, attributes))
            values_str = ", ".join(['?' for _ in attributes])
            values_tuple = [str(v) for v in values]

            sql_statement = 'insert into layers (%s) values (%s)' % (attributes_str, values_str)

            cur = db.execute(sql_statement, values_tuple)

    db.commit()

if __name__ == '__main__':
    models = load_digits_model()

    for m in models:
        print m['net_id']
        for l in m['layers']:
            print "|  ",
            print l.get('layer_name'),
            print l.get('stride')

    # for m in models:
    #     for k, v in get_net_info(m['net_id']).items():
    #         m[k] = v
    #         # break
    #         # print v
    # save_as_sqlite(models)

    # _save_models(models)
