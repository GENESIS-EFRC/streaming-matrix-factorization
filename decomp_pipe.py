import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA, NMF
from rapidz import Stream

_SUPPORT_METHODS = ('nmf', 'pca')

def decomp_op(data, model, **kwargs):
    #m = model(n_components, **kwargs)
    model.fit(data)
    eigen = model.components_
    return eigen, model.transform(data)


def swipping_nmf_loss(ar, comp_ub=5):
    loss = []
    sweeping_grid = range(1, comp_ub + 1, 1)
    for i in sweeping_grid: 
        m = NMF(n_components=i, random_state=23, max_iter=1e3)
        m.fit(ar)
        loss += [m.reconstruction_err_]
    loss = np.asarray(loss)
    return loss


def nmf_ncomp_selection(loss, rtol=1e-3):
    loss = np.asarray(loss)  # cast anyway
    assert loss.ndim == 1
    # find improvement ratio after adding subsequent comp
    imp_ratio = np.abs(np.diff(loss) / loss[:-1])
    inds, = np.where(imp_ratio <= rtol)
    return inds[0] + 1


def configure_nmf_model(data, comp_ub=5, rtol=1e-3, maxiter=1e3):
    loss = swipping_nmf_loss(data, comp_ub)
    n_comp = nmf_ncomp_selection(loss, rtol)
    return NMF(n_components=n_comp, random_state=23, maxiter=maxiter)


def configure_pca_model(data, var_total=0.99, maxiter=1e3):
    return PCA(n_components=var_totl, random_state=23, maxiter=maxiter)


def decomp_pipeline(data, start, method='nmf', cfg_args=(), **kwargs):
    # check first
    if method not in _SUPPORT_METHODS:
        raise ValueError('Only {} are supported currently'.format(_SUPPORT_METHODS))
    # prepare data
    concat_data = data.accumulate(lambda acc, x: acc + [x], start=[])
    start.sink(lambda x: concat_data.state.clear())
    # configure model
    if method == 'nmf':
        model = configure_nmf_model(concat_data, *cfg_args)
    elif method == 'pca':
        model = configure_pca_model(concat_data, *cfg_args)
    # create pipe
    pipe = concat_data.map(decomp_op, model=model, **kwargs)
    components = pipe.pluck(0)
    scores = pipe.pluck(1)
    return locals()


if __name__ == "__main__":
    # Create the streams data goes into
    source = Stream()
    start = Stream()
    method = 'nmf'
    # create the pipeline
    ns = decomp_pipeline(source, start, method=method)

    # plot the data
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
    plt.subplots_adjust(wspace=0.1)
    #fig.tight_layout()

    # Note: funcs have to here as func use global namespace
    def plot_comp(data):
        ax0.cla()
        shift = 0
        for i in range(data.shape[0]):
            ax0.plot(data[i, :] + shift)
            shift += 1.05 * np.max(data[i, :])
        ax0.set_ylabel(r'G$^e$')
        ax0.set_xlabel(r'r ($\mathrm{\AA}$)')


    def plot_weight(data):
        ax1.cla()
        for i, m in zip(range(data.shape[1]), Line2D.filled_markers):
            ax1.plot(data[:, i], marker=m)
        ax1.set_xlabel('Discharge time')
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.tick_right()
        ax1.set_ylabel('Phase ratio')


    # streaming 
    ns["components"].sink(plot_comp)
    ns["scores"].sink(plot_weight)

    # data
    data_ars = np.load('sim_raw_mixed_Gr.npy')

    # Run the data into the pipeline
    for data in data_ars:
        source.emit(data)
        plt.pause(.1)
    #    input()
    plt.show()
