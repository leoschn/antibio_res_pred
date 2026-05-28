import argparse

def load_args_species():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--eval_inter', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--backbone', type=str, default='ResNet18')
    parser.add_argument('--dataset_train_dir', type=str, default='/lustre/fsn1/projects/rech/bun/ucg81ws/synt_class_dataset/eval_1/classification_dataset/split_E')
    parser.add_argument('--dataset_optional_train_dir', type=str,
                        default=None)
    parser.add_argument('--dataset_val_dir', type=str, default='/lustre/fsn1/projects/rech/bun/ucg81ws/synt_class_dataset/eval_1/classification_dataset/split_B')
    parser.add_argument('--dataset_test_dir', type=str, default='/lustre/fsn1/projects/rech/bun/ucg81ws/synt_class_dataset/eval_1/classification_dataset/split_A')
    parser.add_argument('--origin_train', type=str,default='real')
    parser.add_argument('--origin_optional_train', type=str, default='synth')
    parser.add_argument('--origin_val', type=str, default='real')
    parser.add_argument('--origin_test', type=str, default='real')
    parser.add_argument('--save_path', type=str, default='output/best_model_real.pt')
    parser.add_argument('--out_path', type=str, default='output/output_real.csv')
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--n_window', type=int, default=100)
    parser.add_argument('--n_feature', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.)
    parser.add_argument('--wandb', type=str, default='species_prediction_ms2')
    parser.add_argument('--weight', type=str, default='shared')
    args = parser.parse_args()

    return args



