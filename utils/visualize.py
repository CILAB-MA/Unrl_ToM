import matplotlib.pyplot as plt
import os, copy
import numpy as np
from sklearn.manifold import TSNE
import PIL
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class Visualizer:

    def __init__(self):
        self.visualizer = None


    def get_char(self, e_char, epoch, agent_weights=[]):
        model = TSNE(2)
        tsne_results = model.fit_transform(e_char)
        color_palette = ['blue', 'magenta', 'orange', 'limegreen']
        plt.figure()
        preference_index = np.argmax(agent_weights, axis=-1)

        if len(agent_weights) != 0:
            color_list = [color_palette[i] for i in preference_index]
            alpha_list = [weights[i] for i, weights in zip(preference_index, agent_weights)]
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, c=color_list, alpha=alpha_list)
        else:
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5)
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        plt.close('all')
        return pil_image

    def get_power_char(self, e_char, epoch, power_idx, agent_weights=[], saved_model='ATTN'):
        color_palette = ['blue', 'pink']
        plt.figure()
        e_char = np.array(e_char)
        power_idx = np.array(power_idx)
        agent_weights = np.array(agent_weights)
        preference_index = np.argmax(agent_weights, axis=-1)
        powers = ['FRANCE', 'AUSTRIA', 'GERMANY', 'RUSSIA', 'ITALY', 'TURKEY', 'ENGLAND']
        for i in range(7):
            ith_echar = e_char[power_idx==i]
            model = TSNE(2)
            tsne_results = model.fit_transform(ith_echar)
            ith_prefer = preference_index[power_idx == i]
            color_list = [color_palette[i] for i in ith_prefer]
            alpha_list = [weights[i] for i, weights in zip(preference_index, agent_weights)]
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, c=color_list, alpha=alpha_list)
        plt.clf()
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        pil_image.save(f'SAVE_{saved_model}_{powers[i]}_CHAR.jpg', dpi=300)
        plt.close('all')
        return pil_image

    def get_3d_char(self, e_char, epoch, agent_weights=[]):
        model = TSNE(n_components=3, init='pca', random_state=0, perplexity=30, n_iter=5000)
        trans_data = model.fit_transform(e_char)
        color_palette = ['blue', 'magenta', 'orange', 'limegreen']
        fig = plt.figure()
        preference_index = np.argmax(agent_weights, axis=-1)
        color_list = [color_palette[i] for i in preference_index]
        alpha_list = [weights[i] for i, weights in zip(preference_index, agent_weights)]

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(trans_data[:, 0], trans_data[:, 1], trans_data[:, 2], c=color_list, alpha=alpha_list, s=100, marker='+')
        ax.scatter(trans_data[:, 0], trans_data[:, 1], trans_data[:, 2], c=color_list, alpha=alpha_list, s=100, marker='.')


        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title("t-SNE")
        plt.axis('tight')
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        plt.close('all')
        return pil_image
