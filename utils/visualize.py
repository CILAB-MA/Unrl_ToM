import matplotlib.pyplot as plt
import matplotlib
import os, copy
import numpy as np
from sklearn.manifold import TSNE
import PIL
import seaborn as sns
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state

matplotlib.rcParams['contour.linewidth'] = 0

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class Visualizer:

    def __init__(self):
        self.visualizer = None


    def get_char(self, e_char, epoch, agent_weights=[]):
        model = TSNE(2)
        if len(e_char) > 1000:
            e_char = e_char[:1000]
            agent_weights = agent_weights[:1000]
        tsne_results = model.fit_transform(e_char)
        color_palette = ['blue', 'magenta']
        labels = [r'$w_u$', r'$1 - w_u$']  # 'Board','Relation'
        plt.figure()
        preference_index = np.argmax(agent_weights, axis=-1)

        if len(agent_weights) != 0:
            color_list = [color_palette[i] for i in preference_index]  # 1000
            alpha_list = [2 * weights[i] - 1 for i, weights in zip(preference_index, agent_weights)]
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, c=color_list, alpha=alpha_list)
            # unique_preferences = np.unique(preference_index)
            # for pref in unique_preferences:
            #     mask = (preference_index == pref)
            #     x = tsne_results[mask, 0]
            #     y = tsne_results[mask, 1]
            #     # print(x.shape, y.shape, np.array(alpha_list)[mask].shape, color_palette[pref])
            #     sns.kdeplot(x=x, y=y, color=color_palette[pref], fill=True, bw_adjust=.5, alpha=.3)
        else:
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5)
            sns.kdeplot(x=tsne_results[:, 0], y=tsne_results[:, 1], cmap="Blues", fill=True, bw_adjust=.5)

        # 범례 추가 'blue' -> board weight, 'magenta' -> realtion weight
        for i in range(len(color_palette)):
            plt.scatter([], [], color=color_palette[i], label=labels[i])

        plt.legend(loc="upper right")
        # 축 레이블 추가
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        # 제목 추가
        plt.title('Visualization of Unreliability in the Embedding Space')

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
