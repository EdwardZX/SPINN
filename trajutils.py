from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt

# from plotly.graph_objs import *
# import plotly
from fbm import FBM
import torch


def plot_clustering(z_run, labels, engine ='plotly', download = False, folder_name ='clustering'):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """
    def plot_clustering_plotly(z_run, labels):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        trace = Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='PCA on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        trace = Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='tSNE on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download, folder_name):

        labels = labels[:z_run.shape[0]] # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=colors, marker='*', linewidths=0)
        plt.title('PCA on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/pca.png")
        else:
            plt.show()

        plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=colors, marker='*', linewidths=0)
        plt.title('tSNE on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/tsne.png")
        else:
            plt.show()

    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name)


def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]


def v_normalize(x):
    v = np.diff(x,axis=0)
    v_norm = v / (np.linalg.norm(v,axis=0)+1e-6)*np.sqrt(v.shape[0])
    # a = np.linalg.norm(v,axis=0)
    x_0 = np.array([0,0]).reshape(-1,x.shape[1])
    # a = np.std(v_norm,axis=0)
    return np.cumsum(np.vstack([x_0,v_norm]),axis=0),v_norm

def moving_window(x,L,is_header = True, is_Time=False):
    # a = [[x[i:i+seq,:]] for i in range(x.shape[0]-seq)]
    if is_Time:
        x = np.concatenate([np.linspace(0, 1, x.shape[0]).reshape(-1, 1), x], axis=1)
    x_s = np.lib.stride_tricks.sliding_window_view(x,(L,x.shape[1])).squeeze()
    x_0 = np.expand_dims(np.zeros_like(x_s[:,0,:]),axis=1)
    if is_header:
        x_0 = np.expand_dims(np.mean(x_s,axis=1),axis=1)
    output = x_s - x_0
    if is_Time:
        output[:, :, 0] += x_0[:, :, 0]
        x_0[:, :, 0] = 0
    return output, x_0

def moving_window_reverse(X,**kwargs):
    rows = X.shape[0]
    cols = X.shape[1]
    c = X.shape[2]
    L = rows+cols-1
    output = np.zeros((rows+cols-1,c),dtype=np.float64)
    if 'x_0' in kwargs.keys():
        x_0 = kwargs['x_0']
    else:
        x_0 = np.zeros((rows,cols,c),dtype=np.float64)
    X +=x_0
    for i in range(cols):
        output[i:i+rows,:] +=X[:,i,:]

    norm = cols * np.ones_like(output)
    norm[0:cols,:] = np.linspace(1,cols,cols)[:,None].repeat(c,axis=1)
    norm[-cols:, :] = np.linspace(cols, 1, cols)[:, None].repeat(c, axis=1)
    output /= norm
    return output

def concat_traj(X):
    x0 = np.zeros((1,X[0].shape[1]))
    output = []
    for x in X:
        x +=x0
        x0 +=x[-1]
        output.append(x)
    return np.concatenate(output,axis=0)

def generate_x(alpha=[1.],Dt=[1.],time_span=2000,seed=0):
    output = []
    index_output = []
    cnt = seed
    np.random.seed(cnt)
    for alpha_val in alpha:
        fbm_H = alpha_val/2
        f = FBM(time_span - 1, fbm_H)
        for D in Dt:
            np.random.seed(cnt)
            t_x = f.fbm()
            cnt +=1

            np.random.seed(cnt)
            t_y = f.fbm()
            cnt += 1
            x = np.vstack([t_x, t_y]).T
            x, _ = v_normalize(x)
            output.append(D*x)
            index_output.append(np.array([alpha_val,D]).reshape(1,-1))
    return concat_traj(output), index_output

def reject_outliers(data, m=4):
    data[abs(data - np.mean(data)) > m * np.std(data)] = 0
    return data

def cal_autocorrelation_test(x):
        # Tensor (L,B,D)
    N = x.shape[0]
    X = torch.concat((x,torch.zeros(N-1,x.shape[1],x.shape[2])),dim=0)
    # X = x
    fx = torch.fft.fft(X,dim=0)
    ac = torch.fft.ifft(fx * torch.conj(fx),dim=0).real
    return ac[:N]


def save(model,dload,file_name):
    PATH = dload + '/' + file_name
    if os.path.exists(dload):
        pass
    else:
        os.mkdir(dload)
    torch.save(model.state_dict(), PATH)


def load(model,dload,file_name):
    PATH = dload + '/' + file_name
    model.load_state_dict(torch.load(PATH))




if __name__ == '__main__':
    # x = np.random.randn(100,2)
    # x, x_idx = generate_x(alpha=[1e-6],Dt=[1],seed=0)
    # y,x0 = moving_window(x,30)
    # # newy = y+x0
    # newy = moving_window_reverse(y,x_0=x0)
    # # x, x_idx = generate_x(alpha=[0.5,1],Dt=[1],seed=0)
    # plt.plot(x[:,0])
    # plt.show()
    # plt.plot(newy[:, 0])
    # plt.show()
    # a = 1

    # x, x_idx = generate_x(alpha=alpha,Dt=Dt,time_span=time_span,seed=0)

    x, _ = generate_x(alpha=[0.75], Dt=[1], time_span=50 * 51, seed=0)
    X = torch.from_numpy(np.diff(x.reshape(50, 51, 2),axis=1).transpose(1,0,2))
    a = cal_autocorrelation_test(X).cpu().detach().numpy()
    a_sum = np.sum(a,axis=1)
    plt.plot(np.sum(a,axis=1))
    plt.show()


