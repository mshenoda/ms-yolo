import argparse
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import SGD
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter


from utils import create_dataloader, YOLOv1Loss, parse_cfg, build_model

# from torchviz import make_dot

parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument("--cfg", "-c", default="cfg/yolov1.yaml", help="Yolov1 config file path", type=str)
parser.add_argument("--dataset_cfg", "-d", default="cfg/dataset.yaml", help="Dataset config file path", type=str)
parser.add_argument("--weights", "-w", default="", help="Pretrained model weights path", type=str)
parser.add_argument("--output", "-o", default="output", help="Output path", type=str)
parser.add_argument("--epochs", "-e", default=135, help="Training epochs", type=int)
parser.add_argument("--lr", "-lr", default=0.005, help="Training learning rate", type=float)
parser.add_argument("--batch_size", "-bs", default=64, help="Training batch size", type=int)
parser.add_argument("--save_freq", "-sf", default=1, help="Frequency of saving model checkpoint when training", type=int)
parser.add_argument('--tboard', action='store_true', default=False, help='use tensorboard')

args = parser.parse_args()


def train(model, train_loader, optimizer, epoch, device, S, B, train_loss_lst, writer):
    model.train()  # Set the module in training mode
    train_loss = 0
    train_loop = tqdm(train_loader, leave=True)
    for batch_idx, (inputs, labels) in enumerate(train_loop):
        t_start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # back prop
        criterion = YOLOv1Loss(S, B)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        t_batch = time.time() - t_start

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            # fig = plt.figure()
            # inputs = inputs.cpu()  # convert to cpu
            # img_grid = utils.make_grid(inputs)
            # plt.imshow(img_grid.numpy().transpose((1, 2, 0)))
            # plt.savefig(os.path.join(output_path, 'batch0.png'))
            # plt.tight_layout()
            # # plt.show()
            # plt.close(fig)
            inputs = inputs.cpu()  # convert to cpu
            img_grid = utils.make_grid(inputs)
            writer.add_image('image batch0', img_grid, 0)

        # print loss and accuracy
        if batch_idx % 10 == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.1f}%)]  Time: {:.4f}s  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), t_batch, loss.item()))

    # record training loss
    train_loss /= len(train_loader)
    train_loss_lst.append(train_loss)
    return train_loss_lst


def validate(model, val_loader, device, S, B, val_loss_lst):
    model.eval()  # Sets the module in evaluation mode
    val_loss = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = YOLOv1Loss(S, B)
            val_loss += criterion(output, target).item()

    val_loss /= len(val_loader)
    print('Val set: Average loss: {:.4f}\n'.format(val_loss))
    # record validating loss
    val_loss_lst.append(val_loss)
    return val_loss_lst


def test(model, test_loader, device, S, B):
    model.eval()  # Sets the module in evaluation mode
    test_loss = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = YOLOv1Loss(S, B)
            test_loss += criterion(output, target).item()

    # record testing loss
    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    cfg = parse_cfg(args.cfg)
    dataset_cfg = parse_cfg(args.dataset_cfg)
    #img_path, label_path = dataset_cfg['images'], dataset_cfg['labels']
    img_list_path = dataset_cfg['images']
    S, B, num_classes, input_size = cfg['S'], cfg['B'], cfg['num_classes'], cfg['input_size']

    # create output file folder
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    output_path = os.path.join(args.output, 'train', start)
    os.makedirs(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    # build model
    model = build_model(args.weights, S, B, num_classes).to(device)

    # plot model structure
    # graph = make_dot(model(torch.rand(1, 3, args.input_size, args.input_size).cuda()),
    #                  params=dict(model.named_parameters()))
    # graph.render('model_structure', './', cleanup=True, format='png')

    # get data loader
    train_loader, val_loader, test_loader = create_dataloader(img_list_path, 0.8, 0.1, 0.1, args.batch_size,
                                                              input_size, S, B, num_classes)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    # optimizer = Adam(model.parameters(), lr=lr)

    train_loss_lst, val_loss_lst = [], []
    
    print('using tensorboard')
    os.makedirs(output_path, exist_ok=True)
    writer = SummaryWriter(output_path)

    # train epoch
    for epoch in range(args.epochs):
        train_loss_lst = train(model, train_loader, optimizer, epoch, device, S, B, train_loss_lst, writer)
        val_loss_lst = validate(model, val_loader, device, S, B, val_loss_lst)
        writer.add_scalar('Loss/train', np.average(train_loss_lst), epoch)
        writer.add_scalar('Loss/validate', np.average(val_loss_lst), epoch)
        #writer.add_scalar('Loss', {'train':np.average(train_loss_lst),'val':np.average(val_loss_lst)}, epoch)
        # save model weight every save_freq epoch
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(output_path, 'epoch' + str(epoch) + '.pt'))

    test(model, test_loader, device, S, B)

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, 'last.pt'))

    # plot loss, save params change
    fig = plt.figure()
    plt.plot(range(args.epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(args.epochs), val_loss_lst, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, 'loss_curve.jpg'))
    #plt.show()
    plt.close(fig)
