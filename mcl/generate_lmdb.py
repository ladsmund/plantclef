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
    path_infos = os.path.join(path, 'list.txt')
    path_shape = os.path.join(path, 'shape.txt')
    path_mean = os.path.join(path, 'lmdb_mean.txt')
    path_data = os.path.join(path, 'lmdb_data')
    path_labels = os.path.join(path, 'lmdb_labels')

    image_infos = open(path_infos, 'r').read().split('\n')
    image_infos = [i.split() for i in image_infos if len(i) > 1]
    nelements = len(image_infos)

    shape = ast.literal_eval(open(path_shape, 'r').read())

    mag_path = os.path.join(path, 'coefficient_magnitude.npy')
    print (shape[0], nelements)
    
    mag = np.fromfile(mag_path, dtype='float32')
    print "shape shape: ", np.prod((shape[0], nelements))
    print "mag.shape: ", mag.shape
    mag = mag.reshape((shape[0], nelements))
    mean_mag = np.mean(mag,1)
    def normalize(data):
        for i, mn in enumerate(mean_mag):
            if mn > 0:
                data[i,...] /= mn
                data[i,...] -= 1
            data[i,...] /= 2
            data[i,...] *= 128
            data[i,...] += mean_value
            data[i,...] = np.clip(data[i,...], a_min=0, a_max=255)
        return np.uint8(data)
    open(path_mean,'w+').write(str(mean_value))

    # Determine map size
    scat_coeff = read_scat_coeff(image_infos[0][0], shape)
    scat_coeff_serial = serialize_data(scat_coeff, 0)
    datum_size = len(scat_coeff_serial)
    map_size = nelements * datum_size

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
