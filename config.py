import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--eval_inter', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--backbone', type=str, default='ResNet18')
    parser.add_argument('--dataset_train_dir', type=str, default='/lustre/fsn1/projects/rech/bun/ucg81ws/dataset/img_train')
    parser.add_argument('--dataset_val_dir', type=str, default='/lustre/fsn1/projects/rech/bun/ucg81ws/dataset/img_val')
    parser.add_argument('--dataset_test_dir', type=str, default='/lustre/fsn1/projects/rech/bun/ucg81ws/dataset/img_test')
    parser.add_argument('--save_path', type=str, default='output/best_model.pt')
    parser.add_argument('--out_path', type=str, default='output/output.csv')
    parser.add_argument('--label_path', type=str, default='data/antibiores_labels.csv')
    parser.add_argument('--label_col', type=str, default='GEN (mic) cat')
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--wandb', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='ms2')
    args = parser.parse_args()

    return args



