import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DisenSemi Arguments.')
    # parser.add_argument('--target', dest='target', type=int, default=0, help='')
    parser.add_argument('--dataset', dest='dataset', type=str, default='MUTAG')
    parser.add_argument('--n-percents', dest='n_percents', type=int, default=3)
    #parser.add_argument('--multi-target', dest='multi_target', type=str, help='')
    parser.add_argument('--use-unsup-loss', dest='use_unsup_loss', action='store_const', const=True, default=False)
    parser.add_argument('--separate-encoder', dest='separate_encoder', action='store_const', const=True, default=False)
    parser.add_argument('--epoch_select', type=str, default='test_max')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64, help='')
    parser.add_argument('--data_root', type=str, default="data")
    parser.add_argument('--n-factor', dest='n_factor', type=int, default=4, help='')
    parser.add_argument('--n-layer', dest='n_layer', type=int, default=3, help='')
    parser.add_argument('--lr', dest='lr', type=float, default=5e-3,
            help='Learning rate.')
    parser.add_argument('--lamda1', dest='lamda1', type=float, default=1e-4,
            help='Lamda1 rate.')
    parser.add_argument('--lamda2', dest='lamda2', type=float, default=1e-4,
                        help='Lamda2 rate.')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0.00,
            help='')

    # parser.add_argument('--', dest='num_gc_layers', type=int, default=5,
            # help='Number of graph convolution layers before each pooling')
    # parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            # help='')

    return parser.parse_args()
