import torch
import torch.nn as nn
import os
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.data import DataLoader
from Net2d import Net
from loss2d import Loss
import numpy as np
import time
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils.Dataloader2d import Dataset

parser = argparse.ArgumentParser(description='CT_Nodule')

parser.add_argument('--batchSize', type=int, default=64, help='Training batch size')
parser.add_argument('--testBatchSize', type=int, default=64, help='Testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=1, help='Random seed to use. Default=1')
opt = parser.parse_args()


class proj1():
    def __init__(self, config, training_set, testing_set):
        super(proj1, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.CUDA else 'cpu')
        self.model = None
        self.loss = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.training_set = training_set
        self.testing_set = testing_set
        self.train_loss = []
        self.test_loss = []

    def build_model(self):
        loss = Loss(1500)
        self.loss = loss.to(self.device)
        self.model = Net().to(self.device)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40, 60, 80, 100], gamma=0.9)

    def save_checkpoint(self, epoch):
        model_out_path = "checkpoint_axial_implant_2d/" + "model_epoch_{}.pth".format(epoch)
        state = {"epoch": epoch, "model": self.model}
        if not os.path.exists("checkpoint_axial_implant_2d/"):
            os.makedirs("checkpoint_axial_implant_2d/")

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        print(
            '<==============================================================Train '
            'start==============================================================>')
        metrics = []
        start_time = time.time()
        train_loss = 0
        for i, data in enumerate(self.training_set):

            patch, label32, coord32= data
            patch = patch.to(self.device)
            label32 = label32.to(self.device)
            coord32 = coord32.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(patch, coord32)

            loss_output = self.loss(output, label32)
            loss_output[0].backward()

            self.optimizer.step()
            loss_output[0] = loss_output[0].item()
            train_loss += loss_output[0]
            metrics.append(loss_output)
            # print(i, 'Loss_y: {}'.format(loss_y), 'Loss_ini: {}'.format(loss_ini))
        self.train_loss.append(train_loss)
        end_time = time.time()
        metrics = torch.tensor(metrics, device='cpu')
        metrics = np.asarray(metrics, np.float32)
        print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            end_time - start_time))
        print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5])))


    def test(self):
        self.model.eval()
        print(
            '<==============================================================Test '
            'start==============================================================>')
        test_loss = 0

        with torch.no_grad():
            for i, data in enumerate(self.testing_set):
                patch, label32, coord32 = data
                patch = patch.to(self.device)
                label32 = label32.to(self.device)
                coord32 = coord32.to(self.device)

                output = self.model(patch, coord32)


                loss_output = self.loss(output, label32)
                test_loss += loss_output[0].item()
        self.test_loss.append(test_loss)
        print("   Test Average Loss : {}".format(test_loss))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step()
            if epoch % 2 == 0:
                self.save_checkpoint(epoch)

        plt.plot(self.train_loss, label="train_loss")
        plt.plot(self.test_loss, label="test_loss")
        plt.legend()
        plt.show()



def main():
    opt = parser.parse_args()
    print(opt)

    print("===> Loading datasets")
    # Load train set
    train_path = 'axial_implant_2d/'
    train_dataset = Dataset(train_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batchSize)
    test_path = 'axial_implant_2d/'
    test_dataset = Dataset(test_path)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=opt.testBatchSize)
    model = proj1(opt, train_dataloader, test_dataloader)
    model.run()


if __name__ == "__main__":
    main()
