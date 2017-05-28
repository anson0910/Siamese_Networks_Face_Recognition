import numpy as np
import sklearn.datasets


class DataLoader(object):
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, train_data, val_data, img_size, olivetti_faces_data_path):
        self.train_data = train_data
        self.val_data = val_data
        self.olivetti_faces = sklearn.datasets.fetch_olivetti_faces(data_home=olivetti_faces_data_path)
        
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

    def make_oneshot_task(self, num_way):
        """Create pairs of test image, support set for testing num_way way one-shot learning. """
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
        pairs = [test_images, support_set]
        targets = np.zeros((num_way,))
        targets[0] = 1
        return pairs, targets

    def test_oneshot(self, model, num_way, num_trials, verbose=False):
        """Test average num_way way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(num_trials, num_way))
        for i in range(num_trials):
            inputs, targets = self.make_oneshot_task(num_way=num_way)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_correct += 1
        percent_correct = (100.0 * n_correct / num_trials)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct, num_way))
        return percent_correct
