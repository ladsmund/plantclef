import numpy as np
from nets.scatnet_net import process_multiple
import generate_lmdb
import shutil

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('-o', '--output_path', required=True, type=str)
    parser.add_argument('-a', '--nangles', type=int, default=4)
    parser.add_argument('-s', '--scale', type=int, default=3)
    parser.add_argument('-m', '--max_order', type=int, default=3)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    parser.add_argument('-c', '--clean', default=False,
                        action='store_true',
                        help='delete folder with float32 coefficients')
    parser.add_argument('--nimages', type=int, default=None)
    parser.add_argument('--filter_size_factor', type=int, default=2)
    args = parser.parse_args()

    scales = 2 ** np.arange(0, args.scale)

    output_dir_path, output_float_path = process_multiple(
        input_path=args.input_path,
        output_path=args.output_path,
        scales=scales,
        max_order=args.max_order,
        nangles=args.nangles,
        verbose=args.verbose,
        nimages=args.nimages,
        filter_size_factor=args.filter_size_factor)

    generate_lmdb.process(output_dir_path)

    if args.clean:
        shutil.rmtree(output_float_path)