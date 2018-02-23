import matplotlib.pyplot as plt
import numpy as np
import itertools

class Plotter:

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
        plt.xticks(tick_marks, categories, rotation=55)
        plt.yticks(tick_marks, categories)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save:
            plt.savefig(title+'.png', format='png')

        #plt.figure()
        plt.show()


if __name__ == '__main__':
    plotter = Plotter()