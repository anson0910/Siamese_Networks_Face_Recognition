import matplotlib.pyplot as plt, numpy as np
import datasets
from data_loader import DataLoader
from siamese_net import SiameseNet


IMG_SIZE = 105


if __name__ == '__main__':
    lfw_ds = datasets.get_dataset_lfw(path_to_lfw='/home/anson/datasets/lfw', target_size=IMG_SIZE)
    olivetti_ds = datasets.get_dataset_olivetti(path_to_olivetti='/home/anson/datasets/olivetti',
                                                target_size=IMG_SIZE)

    # cut LFW data to training and validation portions
    val_start_idx = int(len(lfw_ds) * 0.8)
    train_data = lfw_ds[:val_start_idx]
    val_data = lfw_ds[val_start_idx:]

    data_loader = DataLoader(train_data=train_data, val_data=val_data, img_size=IMG_SIZE,
                             olivetti_data=olivetti_ds)
    s_net = SiameseNet(data_loader=data_loader, weights_path='weights.h5', input_size=IMG_SIZE)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(olivetti_ds[0, 0, ...], cmap='gray')
    axarr[1].imshow(olivetti_ds[0, 1, ...], cmap='gray')
    plt.waitforbuttonpress()
    img1 = np.reshape(olivetti_ds[0, 0, ...], (1, IMG_SIZE, IMG_SIZE, 1))
    img2 = np.reshape(olivetti_ds[0, 1, ...], (1, IMG_SIZE, IMG_SIZE, 1))
    print(s_net.test(img1, img2))

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(olivetti_ds[0, 0, ...], cmap='gray')
    axarr[1].imshow(olivetti_ds[1, 0, ...], cmap='gray')
    plt.waitforbuttonpress()
    img1 = np.reshape(olivetti_ds[0, 0, ...], (1, IMG_SIZE, IMG_SIZE, 1))
    img2 = np.reshape(olivetti_ds[1, 0, ...], (1, IMG_SIZE, IMG_SIZE, 1))
    print(s_net.test(img1, img2))


