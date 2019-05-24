"""
Argparse helpers
"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args(script, desc=''):
    if script not in ['vis', 'train']:
        raise NotImplementedError('script name = {}'.format(script))

    parser = ArgumentParser(
        description=desc,
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    # Args common to all scripts
    common_parser = parser.add_argument_group('common args')
    common_parser.add_argument('--data_file', default='data/train_x_lpd_5_phr.npz',
                               help='Cleaned/processed LP5 dataset')
    common_parser.add_argument('--exp_dir', default='exp/debug/',
                               help='Cleaned/processed LP5 dataset')
    common_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed')
    common_parser.add_argument('--cuda', action='store_true',
                               help='Use cuda')
    common_parser.add_argument('--debug', action='store_true',
                               help='Load shortened version')

    if script == 'train':
        train_parser = parser.add_argument_group('train args')
        train_parser.add_argument('--hits_only', action='store_true', help='Predict hits only')
        train_parser.add_argument('--activation', default='relu', choices=['swish', 'lrelu', 'relu'],
                                  help='Nonlinear activation in encoders/decoders')
        train_parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of multitrack embeddings')
        train_parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
        train_parser.add_argument('--no_kl', action='store_true', help="Don't use KL (vanilla autoencoder)")
        train_parser.add_argument('--n_tracks', type=int, default=5, help='Number of tracks (between 1 and 5 inclusive)')
        train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
        train_parser.add_argument('--annealing_epochs', type=int, default=20, help='Annealing epochs')
        train_parser.add_argument('--mse_factor', type=float, default=1, help='Weight on mean squared error recon loss')
        train_parser.add_argument('--kl_factor', type=float, default=0.001, help='Constant weight on KL divergence')
        train_parser.add_argument('--max_train', type=int, default=None, help='Maximum training examples to train on')
        train_parser.add_argument('--resume', action='store_true', help='Try to resume from checkpoint')
        train_parser.add_argument('--n_workers', type=int, default=4, help='Number of dataloader workers')
        train_parser.add_argument('--pin_memory', action='store_true', help='Load data into CUDA-pinned memory')
        train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        train_parser.add_argument('--f1_interval', type=int, default=5, help='How often to calcaulte f1 score')
        train_parser.add_argument('--log_interval', type=int, default=100, help='How often to log progress (in batches)')

    args = parser.parse_args()

    # Checks
    if script == 'train':
        if args.n_tracks < 1 or args.n_tracks > 5:
            parser.error('--n_tracks must be between 1 and 5 inclusive')

    return args
