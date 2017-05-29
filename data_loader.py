import numpy as np


class DataLoader(object):
    """Object for loading batches and testing tasks to a siamese net

    Args:
        train_data, val_data (list of numpy 3d arrays):
            training and validation data, 
            where data[i] stores numpy 3d array of size (number of images, image_size, image_size)
        img_size (int): desired image size of loaded data
        olivetti_data (numpy 4d array of size 40 x 10 x image_size x image_size)

    Attributes:
        num_people_train (int): number of different people in training set.
        num_people_val (int): number of different people in validation set.
        img_size (int): size of images

    """

    def __init__(self, train_data, val_data, img_size, olivetti_data):
        self.train_data = train_data
        self.val_data = val_data
        self.olivetti_data = olivetti_data
        
        self.num_people_train = len(train_data)
        self.num_people_val = len(val_data)
        self.img_size = img_size

    def get_training_batch(self, batch_size):
        """Returns training data of size batch_size, where half of instances are same class, half different

        Args:
            batch_size (int): size of batch

        Returns:
            pairs (list of 2 numpy 3d arrays, targets (numpy 1d array): 
                for 0 <= i < batch_size // 2:
                    pairs[0][i, :, :, :] and pairs[1][i, :, :, :] store numpy images of same person
                    targets[i] = 0
                for batch_size // 2 <= i < batch_size:
                    pairs[0][i, :, :, :] and pairs[1][i, :, :, :] store numpy images of different people
                    targets[i] = 1
                    
        """
        person_ids = np.random.choice(self.num_people_train, size=(batch_size,), replace=False)
        pairs = [np.zeros((batch_size, self.img_size, self.img_size, 1)) for _ in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            person_id_1 = person_ids[i]
            num_examples = self.train_data[person_id_1].shape[0]
            idx_1 = np.random.randint(0, num_examples)
            pairs[0][i, :, :, :] = self.train_data[person_id_1][idx_1].reshape(self.img_size, self.img_size, 1)

            # pick images of different class for 1st half, same for 2nd
            person_id_2 = person_id_1 if i >= batch_size // 2 \
                else (person_id_1 + np.random.randint(1, self.num_people_train)) % self.num_people_train
            num_examples = self.train_data[person_id_2].shape[0]
            idx_2 = np.random.randint(0, num_examples)
            pairs[1][i, :, :, :] = self.train_data[person_id_2][idx_2].reshape(self.img_size, self.img_size, 1)
        return pairs, targets

    def get_oneshot_pairs_validation(self, num_way):
        """Returns oneshot pairs from validation set, with only first pair belonging to same person

        Args:
            num_way (int): number of different people in support set 

        Returns:
            test_images (numpy 4d array of size num_way x img_size x img_size x 1):
                num_way copies of one image
            support_set (numpy 4d array of size num_way x img_size x img_size x 1):
                num_way images of different people, with only first image belonging to same person in test_images

        """
        person_ids = np.random.choice(self.num_people_val, size=(num_way,), replace=False)
        indices = np.zeros((num_way,))
        for i, person_id in enumerate(person_ids):
            num_examples = self.val_data[person_id].shape[0]
            indices[i] = np.random.randint(0, num_examples)

        true_person_id = person_ids[0]
        # get 2 indices of images of same person
        idx1, idx2 = np.random.choice(self.val_data[true_person_id].shape[0], replace=False, size=(2,))
        test_images = np.asarray([self.val_data[true_person_id][idx1, :, :]] * num_way).\
            reshape(num_way, self.img_size, self.img_size, 1)

        support_set = np.zeros((num_way, self.img_size, self.img_size))
        support_set[0, :, :] = self.val_data[true_person_id][idx2]
        for i in range(1, num_way):
            person_id = int(person_ids[i])
            idx = int(indices[i])
            support_set[i, :, :] = self.val_data[person_id][idx]
        support_set = support_set.reshape(num_way, self.img_size, self.img_size, 1)
        return [test_images, support_set]

    def get_oneshot_pairs_testing(self):
        """Returns 40 one-shot pairs from test set (olivetti data set), with only first pair belonging to same person

        Returns:
            test_images (numpy 4d array of size num_way x img_size x img_size x 1):
                num_way copies of one image
            support_set (numpy 4d array of size num_way x img_size x img_size x 1):
                num_way images of different people, with only first image belonging to same person in test_images

        """
        person_ids = np.arange(0, 40)
        np.random.shuffle(person_ids)

        true_person_id = person_ids[0]
        # get 2 indices of images of same person
        idx1, idx2 = np.random.choice(10, replace=False, size=(2,))
        test_images = np.asarray([self.olivetti_data[true_person_id, idx1, :, :]] * 40).\
            reshape(40, self.img_size, self.img_size, 1)

        support_set = np.zeros((40, self.img_size, self.img_size))
        support_set[0, :, :] = self.olivetti_data[true_person_id, idx2]
        for i in range(1, 40):
            support_set[i, :, :] = self.olivetti_data[person_ids[i], idx2]
        support_set = support_set.reshape(40, self.img_size, self.img_size, 1)
        return [test_images, support_set]

    def test_oneshot(self, model, data_type, num_way=40, num_trials=50, verbose=False):
        """Test average num_way way one-shot learning accuracy of a siamese neural net over num_trials one-shot tasks
        
        Args:
            model (SiameseNet object): SiameseNet model
            data_type (str): 'val' or 'test' depending on evaluating validation or test set
            num_way (int): number of images of different people to run one-shot trial
            num_trials (int): number of trials to run one-shot trial
            verbose (bool): whether to turn on verbosity mode                         

        Returns:
            Average accuracy of one-shot trials
                    
        """
        correct_count = 0
        if verbose:
            print("Evaluating model on {} one-shot tasks ...".format(data_type))
        for i in range(num_trials):
            inputs = self.get_oneshot_pairs_validation(num_way=num_way) \
                if data_type == 'val' else self.get_oneshot_pairs_testing()
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                correct_count += 1
        percent_correct = (100.0 * correct_count / num_trials)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, num_way))
        return percent_correct
