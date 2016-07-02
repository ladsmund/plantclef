import os
import ast
import numpy as np
import caffe
import lmdb
import time

DEFAULT_MEAN_VALUE = 64

def get_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_scat_coeff(path, shape):
    return np.fromfile(path, dtype='float32').reshape(shape)


def serialize_data(scat_coeff, label):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = scat_coeff.shape[0]
    datum.height = scat_coeff.shape[1]
    datum.width = scat_coeff.shape[2]
    datum.data = scat_coeff.tobytes()
    datum.label = label
    return datum.SerializeToString()


def serialize_labels(label):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels, datum.height, datum.width = [1, 1, 1]
    datum.float_data.extend([int(label)])
    return datum.SerializeToString()


def process(path, mean_value = DEFAULT_MEAN_VALUE):
    path_data = os.path.join(path, 'lmdb_data')
    path_labels = os.path.join(path, 'lmdb_labels')

    path_infos = os.path.join(path, 'list.txt')
    image_infos = open(path_infos, 'r').read().split('\n')
    image_infos = [i.split() for i in image_infos if len(i) > 1]
    nelements = len(image_infos)

    path_shape = os.path.join(path, 'shape.txt')
    shape = ast.literal_eval(open(path_shape, 'r').read())

    channel_norm_factor_path = os.path.join(path, 'channel_norm_factor.npy')
    channel_norm_factor = np.fromfile(channel_norm_factor_path, dtype='float32')
    print "channel_norm_factor.shape: ", channel_norm_factor.shape
    def normalize(data):
        data = data.copy()
        for i, v in enumerate(channel_norm_factor):
            data[i,...] *= v
        return np.uint8(255*data)

    # Normalize and save mean coefficients
    mean_coefficients_path = os.path.join(path, 'mean_coefficients.npy')
    mean_coefficients = np.fromfile(mean_coefficients_path, dtype='float32').reshape(shape)
    lmdb_mean_coeff_path = os.path.join(path, 'lmdb_mean_coeff.npy')
    lmdb_mean_coeff = normalize(mean_coefficients)
    lmdb_mean_coeff.tofile(lmdb_mean_coeff_path)

    # Determine map size
    datum_size = len(serialize_data(lmdb_mean_coeff, 0))
    map_size = nelements * datum_size

    # Initialize LMDB data bases
    env_data = lmdb.open(path_data, map_size=10 * map_size)
    env_labels = lmdb.open(path_labels)
    txn_data = env_data.begin(write=True)
    txn_labels = env_labels.begin(write=True)

    t0 = time.time()
    for i, [path_scat, label] in enumerate(image_infos):
        if not i % 100:
            print "%i: %.1f" % (i, i / (time.time() - t0))

        id = get_id_from_path(path_scat)
        str_id = bytes(id)

        scat_coeff = read_scat_coeff(image_infos[0][0], shape)
        scat_coeff_serial = serialize_data(normalize(scat_coeff), 0)
        label_serial = serialize_labels(label)

        txn_data.replace(str_id.encode('ascii'), scat_coeff_serial)
        txn_labels.replace(str_id.encode('ascii'), label_serial)

    txn_data.commit()
    txn_labels.commit()
    env_data.close()
    env_labels.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path')

    args = parser.parse_args()

    process(path=args.path)
