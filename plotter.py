import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import svm
from sklearn.datasets import make_blobs

class Ploter:

    np.set_printoptions(precision=2)

    def plot_and_show_confusion_matrix(cm, categories, normalize=False,
                                       title='Confusion matrix',
                                       cmap=plt.cm.Blues, save=False):
        """
        Prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`
        :param categories: categories defined in Categories(Enum)
        :param normalize: values between 0 and 1
        :param title: title of the resulting plot
        :param cmap: choose a colormap
        :param save: saves the plot to file (PNG)
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(categories))
        plt.xticks(tick_marks, categories, rotation=45)
        plt.yticks(tick_marks, categories)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save:
            plt.savefig(title+'.png', format='png')

        #plt.figure()
        plt.show()

    def plot_sdg_result(self): #TODO
        # we only take the first two features. We could
        # avoid this ugly slicing by using a two-dim dataset
        colors = "bry"

        # shuffle
        idx = np.arange(counts.shape[0])
        np.random.seed(13)
        np.random.shuffle(idx)
        counts = counts[idx]
        targets = targets[idx]

        # standardize
        print(type(counts))
        mean = counts.mean(axis=0)
        std = counts.std(axis=0)
        counts = (counts - mean) / std

        h = .02  # step size in the mesh

        # create a mesh to plot in
        x_min, x_max = counts[:, 0].min() - 1, counts[:, 0].max() + 1
        y_min, y_max = counts[:, 1].min() - 1, counts[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Put the result into a color plot
        predictions = predictions.reshape(xx.shape)
        cs = plt.contourf(xx, yy, predictions, cmap=plt.cm.Paired)
        plt.axis('tight')

        # Plot also the training points
        for i, color in zip(classifier.classes_, colors):
            idx = np.where(targets == i)
            plt.scatter(counts[idx, 0], counts[idx, 1], c=color,
                        label=targets[i].values,
                        cmap=plt.cm.Paired, edgecolor='black', s=20)
        plt.title("Decision surface of multi-class SGD")
        plt.axis('tight')

        # Plot the three one-against-all classifiers
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        coef = classifier.coef_
        intercept = classifier.intercept_

        def plot_hyperplane(c, color):
            def line(x0):
                return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

            plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                     ls="--", color=color)

        for i, color in zip(classifier.classes_, colors):
            plot_hyperplane(i, color)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    plotter = Ploter()