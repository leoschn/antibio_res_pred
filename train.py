import os

import torch
from torch import nn, optim
import wandb
from dataset.dataset_antibiores import Antibio_Dataset
from model import Classification_model_ms1
import torch.utils.data

def save_model(model, path):
    print('Model saved')
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))


def train(model, data_train, optimizer, loss_function, epoch):
    model.train()

    losses = 0.
    acc = 0.

    for im, label in data_train:
        label = label.long()
        if torch.cuda.is_available():
            im = im.cuda()
            label = label.cuda()
        pred_logits = model.forward(im)
        pred_class = torch.argmax(pred_logits,dim=1)
        acc += (pred_class==label).sum().item()
        loss = loss_function(pred_logits,label)
        losses += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = losses/len(data_train.dataset)
    acc = acc/len(data_train.dataset)
    print('Train epoch {}, loss : {:.3f} acc : {:.3f}'.format(epoch,losses,acc))
    return losses, acc

def test(model, data_test, loss_function, epoch):
    model.eval()
    losses = 0.
    acc = 0.
    for param in model.parameters():
        param.requires_grad = False

    for im, label in data_test:
        label = label.long()
        if torch.cuda.is_available():
            im = im.cuda()
            label = label.cuda()
        pred_logits = model.forward(im)
        pred_class = torch.argmax(pred_logits,dim=1)
        acc += (pred_class==label).sum().item()
        loss = loss_function(pred_logits,label)
        losses += loss.item()
    losses = losses/len(data_test.dataset)
    acc = acc/len(data_test.dataset)
    print('Test epoch {}, loss : {:.3f} acc : {:.3f}'.format(epoch,losses,acc))
    return losses,acc


def run(args):
    #load data

    data_train = Antibio_Dataset(root=args.data_train,label_path=args.label_path,label_col=args.label_col)
    data_val = Antibio_Dataset(root=args.data_val,label_path=args.label_path,label_col=args.label_col)
    data_test = Antibio_Dataset(root=args.data_test,label_path=args.label_path,label_col=args.label_col)
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size)
    data_loader_val = torch.utils.data.DataLoader(data_val, batch_size=args.batch_size)
    data_loader_test =  torch.utils.data.DataLoader(data_test, batch_size=args.batch_size)

    #load model

    model = Classification_model_ms1(backbone = args.backbone, n_class=2)

    model = model.to(torch.float32)
    #load weight
    if args.pretrain_path is not None :
        load_model(model,args.pretrain_path)
    #move parameters to GPU
    if torch.cuda.is_available():
        model = model.cuda()

    if args.wandb :
        with open('wdb_key.txt', 'r') as f:
            key = f.readline().strip()
        os.environ["WANDB_API_KEY"] = key

        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project='antibio_classification')
    #init accumulators
    best_acc = 0
    #init training
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #train model
    for e in range(args.epoches):
        loss, acc = train(model,data_loader_train,optimizer,loss_function,e)
        if args.wandb :
            wandb.log({'epoch':e,'loss_train':loss,'acc_train':acc})
        if e%args.eval_inter==0 :
            loss, acc = test(model,data_loader_val,loss_function,e)
            if args.wandb :
                wandb.log({'epoch':e,'loss_val':loss,'acc_val':acc})
            if acc > best_acc :
                save_model(model,args.save_path)
                best_acc = acc
    load_model(model,args.save_path)
    loss, acc = test(model, data_loader_test, loss_function, e)
    print('Test accuracy : {:.3f}'.format(best_acc))
    if args.wandb :
        wandb.log({'loss_test':loss,'acc_test':acc})

    if args.wandb :
        wandb.finish()
    # plot and save training figs
    #load and evaluate best model
    # load_model(model, args.save_path)
    # os.remove(args.save_path)
