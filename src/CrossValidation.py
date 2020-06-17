'''
Paper Title: Road Surface Recognition Based on DeepSense Neural Network using Accelerometer Data
Created by ITS Lab, Institute of Computer Science, University of Tartu
'''

import numpy as np
import preprocessing, utils


class CV:

    def __init__(self, k=10, network=None):
        """
        Create instance for performing k-fold cross-validation

        :param k: num of folds
        :param network: DeepSense network instance
        """
        # to change the training configurations,
        # set up in network object
        self.network = network
        self.k = k
        self.train = None
        self.labels = None
        self.labels_origin = None
        self.folds = None
        self.cv_scores = []
        self.train_hist = []
        return

    def create_folds(self, shuffle=True, random_state=53):
        """
        Create folds

        :param shuffle: shuffle data before creating folds
        :param random_state: for reproduction
        """
        # check network availability
        if self.network is None:
            print('No network found!')
            return
        # get train features & labels
        self.train = self.network.features
        self.labels_origin = self.network.labels_origin
        self.labels = self.network.labels
        # use all data in train test, this data will be split later in the folds
        if self.network.val_features is not None:
            self.train = np.concatenate([self.train, self.network.val_features])
            self.labels_origin = np.concatenate([self.labels_origin, self.network.val_labels_origin])
            self.labels = np.concatenate([self.labels, self.network.val_labels])
        # check train set availability
        if self.train is not None and self.labels is not None:
            self.folds = preprocessing.make_data_kfold(self.train, self.labels_origin, self.k,
                                                       shuffle=shuffle, random_state=random_state)
        else:
            print('No training data found in network!')
        return

    def train_on_cv(self, checkpoint_callback=True, earlystop_callback=False,
                    csv_callback=False, patience=40, ckpt_period=1,
                    shuffle=True, epoch=1000, batch_size=128,
                    plateau_callback=False, plateau_patience=50,
                    plateau_fractor=0.8):
        """
        Start training on folds

        :param checkpoint_callback: whether to use checkpoint callback
        :param earlystop_callback: whether to use earlystop callback
        :param csv_callback: whether to save training results into a csv file
        :param patience:  patience value for early stop monitor
        :param ckpt_period: number of epochs to save checkpoint
        :param shuffle: shuffle the data before each epoch
        :param epoch: epoches to train
        :param batch_size: batch size used in training, if give 0, full-batch will be used
        :param plateau_callback: whether to use reduce leaning rate on plateau
        :param plateau_patience: epoches to wait before reducing learnning rate
        :param plateau_fractor: factor by which the learning rate will be reduced. new_lr = lr * factor
        """
        # check folds created
        if self.folds is None:
            print('Have not created folds yet.')
            return

        name = self.network.name

        # start k fold validation
        for i, (i_train, i_test) in enumerate(self.folds):
            # prepare dataset
            train = self.train[i_train]
            labels = self.labels[i_train]
            val_train = self.train[i_test]
            val_labels = self.labels[i_test]
            val_labels_origin = self.labels_origin[i_test]
            validation_data = ([val_train, np.full((val_train.shape[0], 1, 1), self.network.WIN_T)], val_labels)
            test_name = name + '_cv%d' % (i+1)
            self.network.name = test_name
            ckpt_path = './out/checkpoint/%s.h5' % test_name
            print('---------------------------------------------------------------------------------')
            print('  Start Cross Validation, Fold %d' % (i+1))
            print('---------------------------------------------------------------------------------')

            # create callbacks
            callbacks = self.network.create_callbacks(checkpoint_callback=checkpoint_callback,
                                                      earlystop_callback=earlystop_callback,
                                                      csv_callback=csv_callback,
                                                      patience=patience, ckpt_period=ckpt_period,
                                                      monitor='val_acc',
                                                      plateau_callback=plateau_callback,
                                                      plateau_fractor=plateau_fractor,
                                                      plateau_monitor='val_loss',
                                                      plateau_patience=plateau_patience,
                                                      ckpt_path=ckpt_path)

            # clean up models & reset
            self.network.reset_model()

            # start kth fold
            hist = self.network.model.fit([train, np.full((train.shape[0], 1, 1), self.network.WIN_T)],
                                          labels, batch_size=batch_size, epochs=epoch, callbacks=callbacks,
                                          validation_data=validation_data, shuffle=shuffle)
            self.train_hist.append(hist.history)

            # restore best validation score
            self.network.model.load_weights(ckpt_path)

            # evaluate kth fold
            print('************************************************************************')
            scores = []
            score = self.network.model.evaluate(validation_data[0], validation_data[1])
            scores.append(score[1]*100)
            # evaluate each category in kth fold
            for i, c in enumerate([0,1,2]):
                mask = np.arange(val_labels_origin.shape[0])[val_labels_origin==c]
                train_c, labels_c = val_train[mask], val_labels[mask]
                score = self.network.model.evaluate([train_c, np.full((train_c.shape[0], 1, 1),
                                                                      self.network.WIN_T)],
                                                    labels_c)
                scores.append(score[1]*100)

            self.cv_scores.append(scores)
            print('************************************************************************')

        # print overall results
        self.print_cv_results()
        return

    def print_cv_results(self):
        """
        Print accuracy performance after cross-validation
        """
        if len(self.cv_scores) == 0:
            print('You should start cross validation first!')
            return

        scores = np.array(self.cv_scores, dtype=np.float64)
        mean = np.mean(scores[:,0])
        std = np.std(scores[:,0])
        print('Cross Validation Results: %.2f%% (+/- %.2f%%)' % (mean, std))
        for c in [0,1,2]:
            print('Category 1: %.2f%% (+/- %.2f%%)' % (np.mean(scores[:,c+1]), np.std(scores[:,c+1])))
        return

    def save_results(self):
        """
        Save cross-validation results including folds, performance, training history
        """
        basename = './out/DeepSense_%s'
        kfolds = '%dfolds.npz' % self.k
        kscore = 'f%d_scores.npz' % self.k
        khist = 'f%d_hist.npz' % self.k
        np.savez_compressed(basename % kfolds, data=self.folds)
        np.savez_compressed(basename % 'data.npz', train=self.train, labels=self.labels,
                            labels_origin=self.labels_origin)
        np.savez_compressed(basename % kscore, scores=self.cv_scores)
        utils.save_obj(basename % khist, obj=self.train_hist, prefix='./out/')
        return
