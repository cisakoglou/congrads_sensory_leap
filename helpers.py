import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.multitest as multi
import scipy
import seaborn as sns
from pylab import text, figure
import matplotlib.pyplot as plt

import defs

def loadSubjects(congrads_path, filename):
    f = open(congrads_path + filename, 'r')
    ids= f.read().split('\t')[:-1]
    f.close()
    ids = [int(id) for id in ids]
    return ids


def run_controlling(data, ids, ages, sexes, sites, fsiqs, scores_run, no_order):
    data = data[np.where((scores_run!=777)&(scores_run!=999) & (scores_run != 0))]
    ids = ids[np.where((scores_run!=777)&(scores_run!=999) & (scores_run != 0))]
    ages = ages[np.where((scores_run != 777) & (scores_run != 999) & (scores_run != 0))]
    sexes = sexes[np.where((scores_run != 777) & (scores_run != 999) & (scores_run != 0))]
    sites = sites[np.where((scores_run != 777) & (scores_run != 999) & (scores_run != 0))]
    fsiqs = fsiqs[np.where((scores_run != 777) & (scores_run != 999) & (scores_run != 0))]

    scores_run = scores_run[np.where((scores_run!=777)&(scores_run!=999) & (scores_run != 0))]

    scores_run = scores_run[np.where(data[:,0] != 0)]
    data = data[np.where(data[:,0] != 0)]
    ids = ids[np.where(data[:,0] != 0)]
    ages = ages[np.where(data[:, 0] != 0)]
    sexes = sexes[np.where(data[:, 0] != 0)]
    sites = sites[np.where(data[:, 0] != 0)]
    fsiqs = fsiqs[np.where(data[:, 0] != 0)]

    print("#subjects: %d" % len(scores_run))

    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    #make it into dataframe
    Xdt = pd.DataFrame(data=data[0:,0:], columns=['coef_' + str(i) for i in range(1,data.shape[1]+1)])  # 1st row as the column names
    Xdt.shape
    y = pd.DataFrame(data=scores_run, columns=['scores'])
    y.shape

    ages = pd.DataFrame(data=ages, columns=['age'])
    Xdt = Xdt.join(ages)

    sexes = pd.DataFrame(data=sexes, columns=['sex'])
    Xdt = Xdt.join(sexes)

    sites = pd.DataFrame(data=sites, columns=['site'])
    Xdt = Xdt.join(sites)

    fsiqs = pd.DataFrame(data=fsiqs, columns=['fsiq'])
    Xdt = Xdt.join(fsiqs)

    # del unnecessary variables

    Xdt = sm.add_constant(Xdt)

    df = Xdt.join(y)

    pvalues_mr = np.zeros([3 * no_order, 1])
    for i in range(1, data.shape[1] + 1):
        coef_no = 'coef_' + str(i)
        model = smf.ols(formula='scores~ ' + coef_no + '+age+sex+fsiq+C(site)', data=df)
        res = model.fit()
        pvalues_mr[i - 1] = res.pvalues[coef_no]
        #print(res.summary())
    #print("p-values before correction:\n")
    #print(pvalues_mr)
    #print(multi.multipletests(np.transpose(np.squeeze(pvalues_mr)), 0.05, 'holm'))
    [reject, pvals_corrected, _, _] = multi.multipletests(np.transpose(np.squeeze(pvalues_mr)), 0.05, 'holm')

    return (pvals_corrected, reject, ids)


def run_univariate_correlation(data, scores, ids, ages, sexes):

    data = data[np.where((scores!=777)&(scores!=999) & (scores != 0))]
    ids = ids[np.where((scores!=777)&(scores!=999) & (scores != 0))]
    ages = ages[np.where((scores != 777) & (scores != 999) & (scores != 0))]
    sexes = sexes[np.where((scores != 777) & (scores != 999) & (scores != 0))]

    scores = scores[np.where((scores!=777)&(scores!=999) & (scores != 0))]

    scores = scores[np.where(data[:,0] != 0)]
    data = data[np.where(data[:,0] != 0)]
    ids = ids[np.where(data[:,0] != 0)]
    ages = ages[np.where(data[:, 0] != 0)]
    sexes = sexes[np.where(data[:, 0] != 0)]

    print("#subjects: %d" % len(scores))

    corrs = np.zeros(data.shape[1])
    pvalues = np.zeros(data.shape[1])
    for i in range(0,data.shape[1]):
        #print(scipy.stats.pearsonr(data[:, i], scores))
        corrs[i], pvalues[i] = scipy.stats.pearsonr(data[:, i], scores)
        #print(scipy.stats.spearmanr(X[:,i],scores))
    return [ids, corrs,pvalues]

def plot_scatter(X, group, scores, pos, correlation, pvalue, color="b", xlabel_text="", ylabel_text="", fig_title=""):

    X = X[np.where((scores != 777) & (scores != 999) & (scores != 0))]
    group = group[np.where((scores != 777) & (scores != 999) & (scores != 0))]
    scores = scores[np.where((scores != 777) & (scores != 999) & (scores != 0))]
    scores = scores[np.where(X[:, 0] != 0)] 
    group = group[np.where(X[:, 0] != 0)]
    X = X[np.where(X[:, 0] != 0)] 

    f = figure()
    ax = f.add_subplot(111)
    sns.regplot(x=scores, y=np.squeeze(X[:, pos]), color='gray', scatter_kws={'alpha': 0.6}, fit_reg=True)
    sns.regplot(x=scores[np.where(group==1)], y=np.squeeze(X[np.where(group==1), pos]), color='royalblue',
                scatter_kws={'alpha':0.6}, fit_reg=False, label='TD')
    sns.regplot(x=scores[np.where(group == 2)], y=np.squeeze(X[np.where(group == 2), pos]), color='darkkhaki',
                scatter_kws={'alpha': 0.6}, fit_reg=False, label='ASD')
    #f.set(xlabel=xlabel_text, ylabel=ylabel_text)
    plt.xlabel(xlabel_text, fontsize=18)
    plt.ylabel(ylabel_text, fontsize=18)
    plt.xlim((min(scores)-3, max(scores)+3))
    # 0.14 for positive correlation, 0.125 for negative correlation
    text(0.14,0.95,"r=" +str(round(correlation,3)), ha='center', va='center', transform=ax.transAxes, fontsize=19)
    text(0.22, 0.88, "p-value=" + str(round(pvalue, 3)), ha='center', va='center', transform=ax.transAxes, fontsize=19)
    plt.legend(loc = 'upper right')
    f.tight_layout()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=scores,
                                                                         y=X[:,pos])
    #f.savefig(CONGRADS_OUTPUT + xlabel_text + "_" + fig_title + '.png', bbox_inches='tight', pad_inches=0,
    #          dpi=1000)
    return [f,ax, slope, intercept, p_value]


def calculate_spatial_correlation(path_1, path_2, hms):
    import scipy.spatial.distance as distance
    from scipy.stats import spearmanr
    """hms used for roi selection"""
    roi_r_img = nilearn.image.smooth_img(defs.ROIS_RS + 'roi_' +hms + '_adapted_all.nii.gz', fwhm=None)
    roi_r = roi_r_img.get_data()

    con_img = nilearn.image.smooth_img(path_1, fwhm=None)
    con = con_img.get_data()
    con_masked_1 = con[np.where(roi_r != 0)][:, 0]

    con_img = nilearn.image.smooth_img(path_2, fwhm=None)
    con = con_img.get_data()
    con_masked_2 = con[np.where(roi_r != 0)][:, 0]

    print(stats.pearsonr(con_masked_1, con_masked_2)) 
    [r,p] = stats.pearsonr(con_masked_1, con_masked_2)
    return (r,p)

