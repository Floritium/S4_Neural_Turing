import argparse
import os

def get_args():
    # ==== Arguments ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_json', type=str, default='configs/copy.json',
                        help='path to json file with task specific parameters')
    parser.add_argument('--saved_model', default='model_copy.pt',
                        help='path to file with final model parameters')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size of input sequence during training')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='number of training steps')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for rmsprop optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for rmsprop optimizer')
    parser.add_argument('--alpha', type=float, default=0.95,
                        help='alpha for rmsprop optimizer')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 constant for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 constant for adam optimizer')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=bool, default=False)
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='number of evaluation steps')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='directory to store checkpointed models')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='checkpoint interval')

    args = parser.parse_args()

    os.makedirs(args.checkpoint_path, exist_ok=True)

    return args