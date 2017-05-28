import lfw_dataset
from data_loader import DataLoader
from siamese_net import SiameseNet


IMG_SIZE = 105


if __name__ == '__main__':
    lfwds = lfw_dataset.get_dataset(path_to_lfw='/home/anson/datasets/lfw', target_size=IMG_SIZE)
    val_start_idx = int(len(lfwds) * 0.8)
    train_data = lfwds[:val_start_idx]
    val_data = lfwds[val_start_idx:]

    data_loader = DataLoader(train_data=train_data, val_data=val_data, img_size=IMG_SIZE,
                             olivetti_faces_data_path='/home/anson/datasets/olivetti')
    s_net = SiameseNet(data_loader=data_loader, weights_path='weights.h5', input_size=IMG_SIZE)
    s_net.train(starting_batch=2000, num_val_trials=50)
