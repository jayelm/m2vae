"""
Split songs into (1) note repr, (2) length repr
"""

import numpy as np
import sys
from multiprocessing import Pool


def reconstruct(data_processed):
    data = np.zeros_like(data_processed)
    for i in range(data.shape[0]):
        for instr in range(data.shape[4]):
            for bar in range(data.shape[1]):
                for note in range(data.shape[3]):
                    for time in range(data.shape[2]):
                        n = data_processed[i, bar, time, note, instr]
                        if n > 0:
                            data[i, bar, time:time+n, note, instr] = 1
    return data


def process(data):
    durations = np.zeros_like(data)
    for i in range(data.shape[0]):
        track = data[i]
        d = durations[i]
        for instr in range(data.shape[4]):
            for bar in range(data.shape[1]):
                for note in range(data.shape[3]):
                    # This is the loop that matters; time
                    is_note = False
                    n_length = None
                    hit_time = None
                    for time in range(data.shape[2]):
                        n = track[bar, time, note, instr]
                        if n:
                            if not is_note:
                                # Note onset
                                assert hit_time is None
                                assert n_length is None
                                is_note = True
                                hit_time = time
                                n_length = 1
                            else:
                                # Note hold
                                assert hit_time is not None
                                assert n_length is not None
                                n_length += 1
                        else:
                            if is_note:
                                assert hit_time is not None
                                assert n_length is not None
                                # Note release
                                # Assign duration
                                d[bar, hit_time, note, instr] = n_length
                                is_note = False
                                hit_time = None
                                n_length = None
                            else:
                                # Rest hold
                                assert hit_time is None
                                assert n_length is None
                    if is_note:
                        # Note was held till end time
                        d[bar, hit_time, note, instr] = n_length
    return durations


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--verify', action='store_true')

    args = parser.parse_args()

    print("Note: this script requires at least 16GB of memory")

    with np.load('train_x_lpd_5_phr.npz') as f:
        data = np.zeros(f['shape'], np.uint8)
        data[[x for x in f['nonzero']]] = 1

    # Split data into chunks
    sub_data = np.array_split(data, args.n_jobs, axis=0)

    pool = Pool(args.n_jobs)
    sub_data_processed = pool.map(process, sub_data)
    pool.close()
    pool.join()

    data_processed = np.concatenate(sub_data_processed)

    if args.verify:
        assert np.all(reconstruct(data_processed) == data)

    np.savez_compressed('./train_x_lpd_5_phr_durations.npz', data_processed)
    np.savez_compressed('./train_x_lpd_5_phr_durations_debug.npz', data_processed[:10000])
