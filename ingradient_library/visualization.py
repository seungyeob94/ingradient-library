import matplotlib.pyplot as plt
import torch

def deep_supervision_visualization(output, target):
    # output shape = (batch size, deepsupervision size, channel size, x, y, z)
    ds = output.shape[1]
    n_cls = output.shape[2]
    center = output.shape[3] // 2
    for d_i in range(ds):
        print("deepsupervision layer", d_i + 1)
        plt.figure(figsize=(3 * (ds), 3*(n_cls+2)))
        for c_i in range(n_cls):
            plt.subplot(int(str(d_i + 1)+str(n_cls+2)+str(c_i+1)))
            plt.imshow(output[0][d_i][c_i][center].detach().cpu(), cmap = 'gray')
            plt.xlabel("class: " + str(c_i))
        plt.subplot(int(str(d_i + 1)+str(n_cls+2)+str(n_cls + 1)))
        plt.imshow(target[0][center].detach().cpu(), vmin = 0, vmax = n_cls)
        plt.xlabel("ground_truth")
        plt.subplot(int(str(d_i + 1)+str(n_cls+2)+str(n_cls + 2)))
        plt.imshow(torch.argmax(output[0][d_i], dim = 0)[center].detach().cpu(), vmin = 0, vmax = n_cls)
        plt.xlabel("predicted")
        plt.show()


def visualization(output, target):
    # output shape = (batch size, channel size, x, y, z)
    n_cls = output.shape[1]
    center = output.shape[2]//2
    plt.figure(figsize=(3, 3*(n_cls+2)))

    for c_i in range(n_cls):
        plt.subplot(int(str(1)+str(n_cls+2)+str(c_i+1)))
        plt.imshow(output[0][c_i][center].detach().cpu(), cmap = 'gray')
        plt.xlabel("class: " + str(c_i))
    plt.subplot(int(str(1)+str(n_cls+2)+str(n_cls + 1)))
    plt.imshow(target[0][center].detach().cpu(), vmin = 0, vmax = n_cls)
    plt.xlabel("ground_truth")
    plt.subplot(int(str(1)+str(n_cls+2)+str(n_cls + 2)))
    plt.imshow(torch.argmax(output[0], dim = 0)[center].detach().cpu(), vmin = 0, vmax = n_cls)
    plt.xlabel("predicted")
    plt.show()