import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import math
import sklearn


class FeatureVector(object):
    """
    Create feature vector from given string and
    bag of words.
    """

    def __init__(self, text):
        """
        Initialize parameter
        :param text: sring
        string of which feature vector need to b created.
        """
        self.text = text.lower()
        self.target_words = None

    def set_target_words(self, target_words):
        """
        setters
        :param target_words: list
        bag of words
        :return: None
        """
        self.target_words = target_words

    def get_target_words(self):
        """
        getters
        :return: list
        bag of words
        """
        return self.target_words

    def dictionary(self):
        """
        Remove extra components from a string and get a dictionary
        of words from it.
        :return: dictionary
        """
        words_dictionary = self.text.split()
        for word in words_dictionary:
            if re.search('<[^<>]+>', word):
                # Strip all HTML
                # Looks for any expression that starts with < and ends with > and replace
                # it with a space
                words_dictionary[int(words_dictionary.index(word))]=' '
            elif re.search('[0-9]+', word):
                # Handle Numbers
                # Look for one or more characters between 0-9 and replace it with text 'number'
                words_dictionary[int(words_dictionary.index(word))]='number'
            elif re.search('(http|https)://[^\s]*', word):
                # Handle URLS
                # Look for strings starting with http:// or https:// and replace it with text 'httpaddr'
                words_dictionary[int(words_dictionary.index(word))]='httpaddr'
            elif re.search('[^\s]+@[^\s]+', word):
                # Handle Email Addresses
                # Look for strings with @ in the middle and reaplace it with text 'emailaddr'
                words_dictionary[int(words_dictionary.index(word))]='emailaddr'
            elif re.search('[$]+', word):
                # Handle $ sign
                # Look for $ sign and replace it with text 'dollar'
                words_dictionary[int(words_dictionary.index(word))]='dollar'

        special_characters = list(' @$/#.-:&*+=[]?!(){},''">_<;%')
        # To remove all special characters that mentioned in above special_characters list from email text.
        text_string = ' '.join(words_dictionary)
        for char in special_characters:
            text_string.replace(char, '')

        words_dictionary = text_string.split()

        # To remove any non alphanumeric characters
        for word in words_dictionary:
            if re.search('[^a-zA-Z0-9]', word):
                words_dictionary[int(words_dictionary.index(word))]=''

        # Using PorterStemmer to stemming the words
        ps = nltk.stem.PorterStemmer()
        ps_words_dictionary = []
        for word in words_dictionary:
            if len(word) > 0:
                try:
                    new_word = ps.stem(word)
                except:
                    continue
                if new_word not in ps_words_dictionary:
                    ps_words_dictionary.append(new_word)

        return ps_words_dictionary

    def vector(self):
        """
        Vector of given string
        :return: array
        """
        dictionary_vector = np.zeros((1, len(self.target_words)), dtype=np.int16)

        ps_dictionary = self.dictionary()
        for word in ps_dictionary:
            if word in self.target_words:
                # Replace 0 with 1 if word in both list match
                dictionary_vector[0][self.target_words.index(word)] = 1

        return dictionary_vector


def negative_words(negative_text_list):
    """
    Get all negative sentiments words from the all the negative reviews.
    :param negative_text_list: list
    list of all strings
    :return: list
    """
    words = []
    for text in negative_text_list:
        fv = FeatureVector(text)
        words = words + fv.dictionary()

    return words


def word_repetition(words_list):
    """
    To get how much time a single word is repeated in the data.
    :param words_list: list
    list of all the words in negative reviews
    :return: dictionary
    words as keys and their repetitions as their respective values.
    """
    dic = {}
    stop_words = pd.read_csv('stopwords.txt')['words:'].to_list()
    for word in words_list:
        if word not in stop_words:
            if word not in dic.keys():
                dic[word] = 1
            else:
                dic[word] += 1

    return dic


def feature_vectors(texts_list, target_words):
    """
    Create feature of all text in given list
    :param texts_list: list
    list of all the strings
    :param target_words: list
    bag of words
    :return: array
    """
    vec = []
    for text in texts_list:
        fv = FeatureVector(text)
        fv.target_words = target_words
        vec.append(fv.vector()[0])

    return np.array(vec)


class TestAccuracy(object):
    """
    test accuracy of SVM
    """

    def __init__(self, x_training, y_training, x_validation, y_validation):
        """
        Initialize
        :param x_training: array
        :param y_training: array
        :param x_validation: array
        :param y_validation: array
        """
        self.x_train = x_training
        self.y_train = y_training
        self.x_cv = x_validation
        self.y_cv = y_validation
        self.C = None
        self.kernel = None

    def svm_prediction(self, C, kernel, x_test):
        """
        get predictions
        :param C: float
        penalty parameter
        :param kernel: str
        linear in this case
        :param x_test: array
        :return: array
        """
        self.C = C
        self.kernel = kernel
        model = sklearn.svm.SVC(C=self.C, kernel=self.kernel, gamma='auto')
        model.fit(self.x_train, self.y_train)
        return model.predict(x_test)

    def accuracy(self, prediction, y):
        """

        :param prediction: array
        :param y: test/cv given predictions/labels
        :return: float
        """
        err = prediction-y
        return len(err[err == 0])/len(y)

    def svm_accuracy(self, x_test, y_test, C, kernel):
        pred = self.svm_prediction(C, kernel, x_test)
        return self.accuracy(pred, y_test)

    def best_c(self, c_list, kernel):
        """
        test which c value is best for the data set
        :param c_list: list
        list of float
        :param kernel:
        :return: tuple
        tuple of two list- train accuracy, test accuracy
        """
        train_accuracy = []
        test_accuracy = []
        print("Please patience. It could take several minute depends on your system configuration.")
        for c in c_list:
            train_accuracy.append(self.svm_accuracy(self.x_train, self.y_train, c, kernel))
            test_accuracy.append(self.svm_accuracy(self.x_cv, self.y_cv, c, kernel))

        return train_accuracy, test_accuracy


def best_c_analysis(x_train, y_train, x_cv, y_cv, c_list):
    """
    Analyse best c on given data set and negative words
    by seeing its graph
    """
    train_acc, test_acc = TestAccuracy(x_train, y_train, x_cv, y_cv).best_c(c_list, 'linear')
    plt.plot(np.arange(len(c_list)), train_acc, label='Training')
    plt.plot(np.arange(len(c_list)), test_acc, label='Test')
    plt.xlabel('C Index')
    plt.ylabel('Accuracy')
    plt.title('Training/Test accuracies')
    plt.legend(loc='best')
    plt.show()
    return None

from pystat import RelativeFrequency

def test_list(upper_bounds, lower_bounds, repeatation_dictionary):
    """
    to get test negative words list
    upper_bound, lower_bounds: list/array
            upper and lower val for limits
    repeatation_dictionary: dictionay carrying all words
            w.r.t. their repetition as values.
    """
    neg_words = []
    rep_df = pd.DataFrame(repeatation_dictionary, index=[0])
    fig, axes = plt.subplots(nrows=len(upper_bounds), ncols=len(lower_bounds), figsize=(16,16))
    test = 0
    for i in range(len(upper_bounds)):
        for j in range(len(lower_bounds)):
            test += 1
            rep_df1 = rep_df.iloc[0][rep_df.iloc[0]>=upper_bounds[i]]
            rep_df2 = rep_df1[rep_df1<=lower_bounds[j]]
            neg_words.append(list(rep_df2.index))

            rep = np.array(list(rep_df2.values))
            rf = RelativeFrequency(rep)
            data_range = rf.get_custom_data_range(rep.min(), rep.max()+10, 10)
            crf = rf.classification(data_range)
            axes[i][j].plot(np.arange(0, len(list(crf['Frequency']))), list(crf['Frequency']))
            axes[i][j].set_xlabel('Range')
            axes[i][j].set_ylabel('Repetition')
            axes[i][j].set_title('Test: '+str(test)+'; Range: '+str(upper_bounds[i])+"-"+str(lower_bounds[j]))

    plt.tight_layout()
    return neg_words


def best_negative_words(test_list, train_test, y_train, cv_test, y_cv, C, kernel):
    """
    to test best negative words
    test_list: lists of chosen negative words
    """
    print("Please patience. It could take several minute depends on your system configuration.")
    result = []
    for test in test_list:
        x_train = feature_vectors(train_test, test)
        x_cv = feature_vectors(cv_test, test)
        result.append(TestAccuracy(x_train, y_train, x_cv, y_cv).svm_accuracy(x_cv, y_cv, C, kernel))

    return result


class Perceptron(object):
    """
    Perceptron algorithm
    """

    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.theta = None
        self.theta0 = None

    def hinge_loss(self, x, y,current_theta, current_theta0):
        return y*(x.dot(current_theta.transpose()) + current_theta0)

    def perceptron_single_step(self, x, y,current_theta, current_theta0):
        loss = self.hinge_loss(x,y,current_theta,current_theta0)
        if loss <= 0.0:
            new_theta = current_theta + (y*x)
            new_theta0 = current_theta0 + y
        else:
            new_theta = current_theta
            new_theta0 = current_theta0

        return new_theta, new_theta0

    def perceptron(self, iteration):
        self.theta = np.zeros(self.x.shape[1])
        self.theta0 = 0.0
        for t in range(iteration):
            for i in range(self.x.shape[0]):
                self.theta, self.theta0 = self.perceptron_single_step(self.x[i], self.y[i],self.theta, self.theta0)

        return self.theta, self.theta0

    def prediction_step(self, x_step, current_theta, current_theta0):
        hing = x_step.dot(current_theta.transpose()) + current_theta0
        if hing <=0:
            return -1
        else:
            return 1

    def prediction(self, x, theta, theta0):
        y = []
        for i in range(x.shape[0]):
            y.append(self.prediction_step(x[i], theta, theta0))
        return np.array(y)

    def prediction_test(self, x_test,y_test,theta, theta0):
        pred = self.prediction(x_test,theta, theta0)
        test = y_test-pred
        success = test[test==0]
        return len(success)/len(y_test)

    def average_perceptron(self, iteration):
        """
        Average perceptron algorithm
        :param iteration: int
        number of iterations
        :return:
        """
        self.theta = np.zeros(self.x.shape[1])
        self.theta0 = 0.0
        av_theta = []
        av_theta0 = []
        for t in range(iteration):
            for i in range(self.x.shape[0]):
                self.theta, self.theta0 = self.perceptron_single_step(self.x[i], self.y[i],self.theta, self.theta0)
            av_theta.append(self.theta)
            av_theta0.append(self.theta0)

        return np.array(av_theta).mean(axis=0), np.array(av_theta0).mean()


class Pegasos(Perceptron):
    """
    Pegasos Algorithm
    """

    def __init__(self, x_train, y_train, L):
        Perceptron.__init__(self, x_train, y_train)
        self.L = L

    def pegasos_single_step(self, x, y, eta, current_theta,current_theta0):
        hinge_loss = self.hinge_loss(x, y, current_theta, current_theta0)
        if hinge_loss < 1:
            current_theta = (1-eta*self.L)*current_theta + eta*y*x
            current_theta0 = current_theta0 + eta*y
        else:
            current_theta = (1-eta*self.L)*current_theta

        return current_theta, current_theta0

    def pegasos_full(self, T):
        n = self.x.shape[0]
        self.theta = np.zeros(self.x.shape[1])
        self.theta0 = 0.0
        c = 0
        for i in range(T):
            c+=1
            eta = 1/math.sqrt(c)
            for k in range(n):
                self.theta, self.theta0 = self.pegasos_single_step(self.x[k], self.y[k], eta, self.theta, self.theta0)

        return self.theta, self.theta0


def precision_recall_curve(x_train, y_train, x_test, y_test, iteration):
    precision = []
    recall = []
    target_name = ["Negative_sentiments", "positive_sentiments"]
    prcp = Perceptron(x_train, y_train)
    theta, theta0 = prcp.average_perceptron(iteration)

    for i in range(1,x_test.shape[0]):
        true_positive = 0
        false_negative = 0
        false_positive = 0
        pred = prcp.prediction(x_test[:i], theta, theta0)
        for j in range(pred.shape[0]):
            if pred[j] == -1 and y_test[j] == -1:
                true_positive += 1
            elif pred[j] == 1 and y_test[j] == 1:
                false_negative += 1
            elif pred[j] == 1 and y_test[j] == -1:
                false_positive += 1

        try:
            prec = true_positive/(true_positive + false_positive)
        except:
            prec = 0

        try:
            rec = true_positive/(true_positive + false_negative)
        except:
            rec = 0
        precision.append(prec)
        recall.append(rec)

    return precision, recall


def best_perceptron_itr(x_train, y_train, x_test, y_test, itr_list):
    my_predictions = []
    for itr in itr_list:
        print("Please patience. It could take several minute depends on your system configuration.")
        prc = Perceptron(x_train, y_train)
        theta, theta0 = prc.average_perceptron(itr)
        my_predictions.append(prc.prediction_test(x_test, y_test, theta, theta0))

    plt.plot(itr_list, my_predictions, label="My")
    plt.title("Best Perceptron Iteration")
    plt.xlabel("Iterations")
    plt.ylabel("Predictions")
    plt.legend(loc="best")
    plt.show()


def pegasos_best_L(x_train, y_train, L_list, T):
    """
    Tell which value of L provides best prediction.

    L_list: list /array of L values to test
    T: iteration
    """
    train_result = []
    for i in L_list:
        print("Please patience. It could take several minute depends on your system configuration.")
        pg = Pegasos(x_train,y_train,i)
        t,t0=pg.pegasos_full(T)
        train_result.append(pg.prediction_test(x_train,y_train, t, t0))

    plt.plot(np.arange(len(train_result)), train_result, label="Training Predictions")
    plt.legend(loc="best")
    plt.title("Predictions on Lambda-Value")
    plt.xlabel("Lambda-Values")
    plt.ylabel("Predictions")
    plt.show()


def pegasos_best_iteration(x_train, y_train, L, T_list):
    """
    Tells on which itration value pegasos gives best result.

     L: learning rate
     T_list: list/array of number of iteration
    """
    train_result = []
    for i in T_list:
        print("Please patience. It could take several minute depends on your system configuration.")
        pg = Pegasos(x_train,y_train,L)
        t,t0=pg.pegasos_full(i)
        train_result.append(pg.prediction_test(x_train,y_train, t, t0))

    plt.plot(np.arange(len(train_result)), train_result, label="Training Predictions")
    plt.legend(loc="best")
    plt.title("Predictions on Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Predictions")
    plt.show()
