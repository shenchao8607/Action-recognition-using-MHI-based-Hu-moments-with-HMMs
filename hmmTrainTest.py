__author__ = 'calp'

import argparse
import cv2
import warnings

import numpy as np
import scipy.io as sio
from hmmlearn import hmm
from sklearn import preprocessing, metrics
from sklearn.cross_validation import LeaveOneOut
from sklearn.decomposition import FastICA, PCA

import hmm_util

warnings.filterwarnings('ignore')


class VideoRecognizer:
    """
    Video recoginion with HMM
    """

    def __init__(self, args):
        self.predicted = []
        self.expected = []
        self.args = args
        self.model = dict()
        self.fullDataTrainHmm = {}
        self.categories = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2']
        self.persons = ['daria_', 'denis_', 'eli_', 'ido_', 'ira_', 'lena_', 'lyova_', 'moshe_', 'shahar_']

    def extractFeature(self, video):
        """
        extract feature of the video
        :param video: video array
        :return: video feature
        """
        images = []
        counter = 0
        for x in range(0, video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]
            res = cv2.resize(gray, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            if self.args.feature_type == 'Hu':
                hu = cv2.HuMoments(cv2.moments(res)).flatten()
                images.append(hu)
            else:
                images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
            counter += 1
        return images

    def extractClassicMhiFeature(self, video):
        """
        extract the traditional mhi of the video
        :param video: source video
        :return: tradional mhi feature of video
        """
        previous = None
        mhi = None
        images = []
        counter = 0
        for x in range(0, video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]
            if previous is not None:
                silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                mhi = cv2.add(mhi, -15)
                mhi = cv2.addWeighted(silhouette, 1.0, mhi, 1.0, 0)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
                if self.args.feature_type == 'Hu':
                    hu = cv2.HuMoments(cv2.moments(res)).flatten()
                    images.append(hu)
                else:
                    images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
            else:
                mhi = np.zeros(gray.shape, gray.dtype)
            previous = gray.copy()
            counter += 1
        return images

    def extractMhiFeature(self, video):
        """
        extract modified MHI of the video
        :param video: source video
        :return: Modified MHI feature of video
        """
        previous = None
        mhi = None
        images = []
        counter = 0
        for x in range(0, video.shape[2]):
            gray = video[:, :, x]
            gray = gray[5:-5, 10:-10]
            gray = cv2.threshold(gray, 0.5, 255, cv2.THRESH_BINARY)[1]
            if previous is not None:
                silhouette = cv2.addWeighted(previous, -1.0, gray, 1.0, 0)
                mhi = cv2.addWeighted(silhouette, 1.0, mhi, 0.9, 0)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                res = cv2.resize(mhi, None, fx=self.args.resize, fy=self.args.resize, interpolation=cv2.INTER_CUBIC)
                if self.args.feature_type == 'Hu':
                    hu = cv2.HuMoments(cv2.moments(res)).flatten()
                    images.append(hu)
                else:
                    images.append(np.append(res.sum(axis=0), res.sum(axis=1)))
            else:
                mhi = np.zeros(gray.shape, gray.dtype)
            previous = gray.copy()
            counter += 1
        return images

    def loadVideos(self):
        """
        Load the video data, Extract feature and train hmm model
        """
        mat_contents = sio.loadmat('data/original_masks.mat')
        mat_contents = mat_contents['original_masks']
        counter = 0
        for category_name in self.categories:
            """Each  category"""
            images = []
            for person in self.persons:
                """Each person"""
                if person == 'lena_' and (category_name == 'run' or category_name == 'skip' or category_name == 'walk'):
                    """Person is Lena and category run, skip or walk"""
                    video = mat_contents[person + category_name + '1'][0][0]
                    if self.args.mhi:
                        data = self.extractMhiFeature(video)
                    else:
                        data = self.extractFeature(video)
                    images.append(data)
                    video = mat_contents[person + category_name + '2'][0][0]
                    if self.args.mhi:
                        data = self.extractMhiFeature(video)
                    else:
                        data = self.extractFeature(video)
                    images.append(data)
                else:
                    video = mat_contents[person + category_name][0][0]
                    if self.args.mhi:
                        data = self.extractMhiFeature(video)
                    else:
                        data = self.extractFeature(video)
                    images.append(data)
            if images.__len__() != 0:
                loo = LeaveOneOut(images.__len__())
                images = np.array(images)
                """train hmm with category all video"""
                self.fullDataTrainHmm[category_name], std_scale, std_scale1 = self.train(images)
                self.model[category_name] = {}
                self.model[category_name]['hmm'] = []
                self.model[category_name]['std_scale'] = []
                self.model[category_name]['std_scale1'] = []
                self.model[category_name]['data'] = []
                for train, test in loo:
                    markov_model, std_scale, std_scale1 = self.train(images[train])
                    self.model[category_name]['hmm'].append(markov_model)
                    self.model[category_name]['std_scale'].append(std_scale)
                    self.model[category_name]['std_scale1'].append(std_scale1)
                    self.model[category_name]['data'].append(images[test])
            self.target_names = self.categories
            counter += 1

    def train(self, images):
        """
        HMM model train with images data
        :param images: array of feature of images
        :return: hmm, std scale, sta_scale
        """
        scaled_images = []
        length = []
        for file in images:
            scaled_images.extend(file)
            length.append(file.__len__())
        std_scale1 = None
        if self.args.preprocess_method == "PCA":
            std_scale1 = preprocessing.StandardScaler()
            std_scale = PCA(n_components=self.args.decomposition_component, random_state=55)
        elif self.args.preprocess_method == "FastICA":
            std_scale1 = preprocessing.StandardScaler()
            std_scale = FastICA(n_components=self.args.decomposition_component, random_state=55)
        elif self.args.preprocess_method == "Normalizer":
            std_scale = preprocessing.Normalizer()
        else:
            std_scale = preprocessing.StandardScaler()
        if std_scale1 is not None:
            std_scale1.fit(scaled_images)
            scaled_images = std_scale1.transform(scaled_images)
        std_scale.fit(scaled_images)
        scaled_images = std_scale.transform(scaled_images)

        ####################TRAIN#########################
        if self.args.gmm_state_number == 1:
            markov_model = hmm.GaussianHMM(n_components=self.args.state_number, n_iter=10, random_state=55)
        else:
            markov_model = hmm.GMMHMM(n_components=self.args.state_number, n_mix=self.args.gmm_state_number, n_iter=100, random_state=55)
        if self.args.left2Right:
            startprob, transmat = hmm_util.initByBakis(self.args.state_number, 2)
            markov_model.init_params = "cm"
            markov_model.params = "cmt"
            markov_model.startprob_ = startprob
            markov_model.transmat_ = transmat
        markov_model.fit(scaled_images, length)
        return markov_model, std_scale, std_scale1

    def testLoaded(self):
        """
        HMM test and plot confision matrix.
        """
        for category in self.categories:
            """Each category"""
            for loo_index, data1 in enumerate(self.model[category]['data']):
                """Each video"""
                for data in data1:
                    """Each leave one out test set"""
                    if self.model[category]['std_scale1'][loo_index] is not None:
                        data = self.model[category]['std_scale1'][loo_index].transform(data)
                    data = self.model[category]['std_scale'][loo_index].transform(data)
                    for index in range(data.__len__() - self.args.window):
                        """Each subvideo"""
                        image = data[index: index + self.args.window]
                        max = self.model[category]['hmm'][loo_index].score(image)
                        predictedCategory = category
                        for testedCategory in self.categories:
                            """find maximum"""
                            if testedCategory != category:
                                score = self.fullDataTrainHmm[testedCategory].score(image)
                                if score > max:
                                    max = score
                                    predictedCategory = testedCategory
                        self.expected.append(category)
                        self.predicted.append(predictedCategory)
        print("Classification report for classifier \n%s\n" % (metrics.classification_report(self.expected, self.predicted)))
        cm = metrics.confusion_matrix(self.expected, self.predicted)
        print("Confusion matrix:\n%s" % cm)
        hmm_util.plotConfusionMatrix(self.expected, self.predicted, self.target_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature-type', type=str, dest='feature_type', default='Hu', help='Feature type. * "7 Hu" *"Projection" (default: %(default)s)')
    parser.add_argument('-g', '--gmm-state-number', type=int, dest='gmm_state_number', default=1, help='Number of states in the GMM. (default: %(default)s)')
    parser.add_argument('-s', '--state-number', type=int, dest='state_number', default=4, help='Number of states in the model. (default: %(default)s)')
    parser.add_argument('-p', '--preprocess-method', type=str, dest='preprocess_method', default='Normalizer',
                        help='Data preprocess method.* "PCA" *"StandardScaler" *"FastICA" *"Normalizer" (default: %(default)s)')
    parser.add_argument('-dc', '--decomposition-component', type=int, dest='decomposition_component', default=7,
                        help='Principal axes in feature space, representing the directions of maximum variance in the data. '
                             'The components are sorted by ``explained_variance_``. (default: %(default)s)')
    parser.add_argument('-r', '--resize', type=float, dest='resize', default=1, help='Frame resize ratio. (default: %(default)s)')
    parser.add_argument('-w', '--window', type=int, dest='window', default=15, help='Frame window size. (default: %(default)s)')
    parser.add_argument('-l2r', '--left-2-right', type=bool, dest='left2Right', default=True, help='Left to right HMM model. (default: %(default)s)')
    parser.add_argument('-mhi', '--mhi', type=bool, dest='mhi', default=True, help='Do use MHI Feature extraction? (default: %(default)s)')

    args = parser.parse_args()
    videoRecognizer = VideoRecognizer(args)
    videoRecognizer.loadVideos()
    videoRecognizer.testLoaded()
