import argparse
import warnings

from magvit2_pytorch import VideoTokenizer, VideoTokenizerTrainer

parser = argparse.ArgumentParser(description='Magvit Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--data-type', default='videos', type=str,
                    help='dataset type')
parser.add_argument('--load-from-path', default=None, type=str,
                    help='if not None, load from path')
parser.add_argument('--bn', default=1, type=int,
                    help='batchsize')


def main():
    args = parser.parse_args()
    tokenizer = VideoTokenizer(
        image_size = 128,
        init_dim = 64,
        max_dim = 512,
        codebook_size = 1024,
        layers = (
            'residual',
            'compress_space',
            ('consecutive_residual', 2),
            'compress_space',
            ('consecutive_residual', 2),
            'linear_attend_space',
            'compress_space',
            ('consecutive_residual', 2),
            'attend_space',
            'compress_time',
            ('consecutive_residual', 2),
            'compress_time',
            ('consecutive_residual', 2),
            'attend_time',
        )
    )

    trainer = VideoTokenizerTrainer(
        tokenizer,
        dataset_folder = args.data,     # folder of either videos or images, depending on setting below
        dataset_type = args.data_type,                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
        batch_size = args.bn,
        grad_accum_every = 8,
        learning_rate = 2e-5,
        num_train_steps = 1_000_000,
        optimizer_kwargs={"betas": (0.9, 0.99)}, # From the paper
        use_wandb_tracking = True,
        load_from_path=args.load_from_path
    )
    trainer.train()


if __name__ == '__main__':
    main()

