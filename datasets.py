import pickle, os, cv2, sklearn.datasets, numpy as np


def scale_crop(img, crop_factor=1.0):
    """Utility to take a bbox and image bounds and scale the bbox by crop_factor

    Args:
        img (numpy 2d array): image to crop
        crop_factor (float): scale of cropping

    Returns:
        cropped image (numpy 2d array)

    """
    img_height, img_width = img.shape
    x, y, w, h = 0, 0, img_width, img_height
    
    x = x + w // 2 - w // 2 * crop_factor
    y = y + h // 2 - h // 2 * crop_factor
    w = w * crop_factor
    h = h * crop_factor

    if (x + w) > img_width:
        x -= ((x + w) - img_width)
    if (y + h) > img_height:
        y -= ((y + h) - img_height)

    x, y, w, h = map(int, (x, y, w, h))

    x = np.clip(x, 0, img_width - w)
    y = np.clip(y, 0, img_height - h)

    return img[y:y + h, x:x + w]


def get_dataset_lfw(path_to_lfw, target_size, min_imgs_per_person=2):
    """Returns LFW data

    Args:
        path_to_lfw (str): path to LFW dataset folder
        target_size (int): desired image size of resized images
        min_imgs_per_person (int): minimum image files of each person in dataset to keep

    Returns:
        data (list of numpy 3d arrays): 
            where data[i] stores numpy 3d array of size (number of images, target_size, target_size) 

    """
    lfw_pickle_filename = os.path.join(path_to_lfw, 'lfw_pickle.p')
    if os.path.exists(lfw_pickle_filename):
        return pickle.load(open(lfw_pickle_filename, 'rb'))

    data = []
    if not os.path.exists(path_to_lfw):
        print('Cannot find dataset at %s' % path_to_lfw)
        return

    print('Cannot find pickle file of LFW, constructing...')
    for people_dir in os.listdir(path_to_lfw):
        num_imgs = len(os.listdir(os.path.join(path_to_lfw, people_dir)))
        if num_imgs >= min_imgs_per_person:
            imgs = np.zeros((num_imgs, target_size, target_size))
            for idx, img_filename in enumerate(os.listdir(os.path.join(path_to_lfw, people_dir))):
                img = cv2.imread(os.path.join(path_to_lfw, people_dir, img_filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cropped_img = scale_crop(img, crop_factor=0.4)
                cropped_img = cv2.resize(cropped_img, (target_size, target_size))
                imgs[idx, ...] = cropped_img
            data.append(imgs)

    pickle.dump(data, open(lfw_pickle_filename, 'wb'))
    return data


def get_dataset_olivetti(path_to_olivetti, target_size):
    """Returns olivetti data

    Args:
        path_to_olivetti (str): path to olivetti dataset folder
        target_size (int): desired image size of resized images

    Returns:
        data (numpy 4d array of size 40 x 10 x target_size x target_size)
    """
    olivetti_pickle_filename = os.path.join(path_to_olivetti, 'olivetti_pickle.p')
    if os.path.exists(olivetti_pickle_filename):
        return pickle.load(open(olivetti_pickle_filename, 'rb'))

    print('Cannot find pickle file of olivetti, constructing...')
    olivetti_imgs = sklearn.datasets.fetch_olivetti_faces(data_home=path_to_olivetti)['images']
    data = np.zeros((40, 10, target_size, target_size))

    for i, img in enumerate(olivetti_imgs):
        person_id = i // 10
        idx = i % 10
        resized_img = cv2.resize(img, (target_size, target_size))
        data[person_id, idx, ...] = resized_img

    pickle.dump(data, open(olivetti_pickle_filename, 'wb'))
    return data
