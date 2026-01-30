import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--eval_inter', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--backbone', type=str, default='ResNet18')
    parser.add_argument('--dataset_train_dir', type=str, default='data/img_ms1/train_data')
    parser.add_argument('--dataset_val_dir', type=str, default='data/img_ms1/val_data')
    parser.add_argument('--dataset_test_dir', type=str, default='data/img_ms1/test_data')
    parser.add_argument('--save_path', type=str, default='output/best_model.pt')
    parser.add_argument('--out_path', type=str, default='output/output.csv')
    parser.add_argument('--label_path', type=str, default='data/antibiores_labels.csv')
    parser.add_argument('--label_col', type=str, default='GEN (mic) cat')
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--wandb', type=str, default=None)
    args = parser.parse_args()

    return args



