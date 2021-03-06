import os
import ast
import numpy as np
import caffe
import lmdb
import time
import PIL.Image
from datetime import datetime

DEFAULT_MEAN_VALUE = 64


def _save_image(image, filename):
    # converting from BGR to RGB
    image = image[[2, 1, 0], ...]  # channel swap
    # convert to (height, width, channels)
    image = image.astype('uint8').transpose((1, 2, 0))
    image = PIL.Image.fromarray(image)
    image.save(filename)


def _save_mean(mean, filename):
    """
    Saves mean to file

    Arguments:
    mean -- the mean as an np.ndarray
    filename -- the location to save the image
    """
    if filename.endswith('.binaryproto'):
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels = mean.shape[0]
        blob.height = mean.shape[1]
        blob.width = mean.shape[2]
        blob.data.extend(mean.astype(float).flat)
        with open(filename, 'wb') as outfile:
            outfile.write(blob.SerializeToString())

    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        _save_image(mean, filename)
    else:
        raise ValueError('unrecognized file extension')


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


def process(path, mean_value=DEFAULT_MEAN_VALUE):
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
            data[i, ...] *= v
        return np.uint8(255 * data)

    # Normalize and save mean coefficients
    mean_coefficients_path = os.path.join(path, 'mean_coefficients.npy')
    mean_coefficients = np.fromfile(mean_coefficients_path, dtype='float32').reshape(shape)
    lmdb_mean_coeff = normalize(mean_coefficients)
    _save_mean(lmdb_mean_coeff, os.path.join(path, 'lmdb_mean_coeff.binaryproto'))
    _save_mean(lmdb_mean_coeff, os.path.join(path, 'lmdb_mean_coeff.png'))

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
        if not i % 1000:
            print "%s: %i: %.1f" % (str(datetime.now()), i, i / (time.time() - t0))

        id = get_id_from_path(path_scat)
        str_id = bytes(id)

        scat_coeff = read_scat_coeff(path_scat, shape)
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
