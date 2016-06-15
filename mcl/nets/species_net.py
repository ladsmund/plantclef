import os
from caffe import layers as L
from caffe_net import caffenet

DEFAULT_DATA_PATH = os.path.join(os.getcwd(), 'data')


def get_data_layer(source, mirror):
    transform_param = dict(mirror=mirror, crop_size=227)
    species_data, species_label = L.ImageData(transform_param=transform_param,
                                              source=source,
                                              batch_size=50,
                                              new_height=256,
                                              new_width=256,
                                              ntop=2)
    return species_data, species_label


def speciesnet(train=True, learn_all=False, source=None, num_classes=None, **kwargs):
    if source is None:
        file_name = 'train.txt' if train else 'test.txt'
        source = os.path.join(DEFAULT_DATA_PATH, file_name)

    if num_classes is None:
        with open(os.path.join(DEFAULT_DATA_PATH, 'species.txt')) as f:
            species = f.readlines()
            num_classes = len(species)

    species_data, species_label = get_data_layer(source, train)

    return caffenet(data=species_data, label=species_label, train=train,
                    num_classes=num_classes,
                    classifier_name='fc8_plant',
                    learn_all=learn_all,
                    **kwargs)
