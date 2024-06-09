import argparse
import argcomplete

def init_arguments():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--seed', type=int, default=1000, help="Seed value for RNGs")
    parser.add_argument('--task', action='store', choices=list(["copy", "repeat-copy", "seq-mnist"]), default='copy',
                        help="Choose the task to train (default: copy)")
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help="Checkpoint interval (default: {}). "
                             "Use 0 to disable checkpointing".format(10))
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='directory to store checkpointed models')
    parser.add_argument('--report_interval', type=int, default=10,
                        help="Reporting interval")
    parser.add_argument('--log', action='store_true', help="Enable logging")

    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    return args