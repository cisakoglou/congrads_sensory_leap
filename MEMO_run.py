import os,subprocess,glob, sys
import statsmodels

import pandas as pd
import nilearn
import numpy as np
import nibabel as nib
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# more specialized
import statsmodels.stats.multitest as multi

import defs
import helpers

import utils
import utils_stats as ustats
import study_utils as su


## run congrads + create list of subjects that should be included

## create average connectopy, invert if needed

## reorder and invert individual connectopies, when needed

##
clinical_info = pd.read_excel(defs.clinical_info_file)
sbj_ids_included = utils.loadSubjects(defs.CONGRADS_OUTPUT_HARIRI, 'sbj_ids_included.txt')
print(len(sbj_ids_included))

## export model selection figure

## association with vineland daily-living scatterplot
no_order = 6
hms = 'left'

sbjs = sbj_ids_included.copy()

ids = np.zeros(len(sbjs))
ages = np.zeros(len(sbjs))
sexes = np.zeros(len(sbjs))
group = np.zeros(len(sbjs))
sites = np.zeros(len(sbjs))
fsiqs = np.zeros(len(sbjs))

X = np.zeros([len(sbjs), 3*no_order])
uni_scores = np.zeros([len(sbjs), 22])

coef_labels = {0: "x", 1: "y", 2: "z",
               3: "x2", 4: "y2", 5: "z2",
               6: "x3", 7: "y3", 8: "z3",
               9: "x4", 10: "y4", 11: "z4",
               12: "x5", 13: "y5", 14: "z5",
               15: "x6", 16: "y6", 17: "z6"}
#define dictionaries for the names of the scores as will be printed
uni_scores_names = {4: 'SRS-2', #Social Responsiveness Scale-2 (SRS-2) Total T-score (combined parent- and self-report)
                    5: 'SSP', #Total Short Sensory Profile (SSP) score (parent-report)
                    6: 'RBS-R', #Repetitive Behaviors Scale-Revised (RBS-R) Total score (paren-report)
                    7: 'ADI-R Social', # total score
                    8: 'ADI-R Comm.', # total score
                    9: 'ADI-R RBS', #total domain score
                    18: 'Vineland-II Comm.',
                    19: 'Vineland-II D.Liv.',
                    20: 'Vineland-II Soc.',
                    21: 'Vineland-II ABC'} #Adaptive Behaviour Composite (ABC) standard score
for count, sbj_id in enumerate(sbjs):
    tsm_file = defs.CONGRADS_OUTPUT_HARIRI + str(
        sbj_id) + '/TSM_' + hms + '/Order_' + str(no_order) + '/roi_' + hms + '_inverted.cmaps.tsm.trendcoeff.txt'
    tsm1 = pd.read_csv(tsm_file, sep=" ", header=None).values[0][0]
    ids[count] = sbj_id
    ages[count] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_age.values[0]
    sexes[count] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_sex.values[0]
    sites[count] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_site.values[0]
    fsiqs[count] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_fsiq.values[0]
    group[count] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_diagnosis.values[0]  # control=1, asd=2

    X[count, :] = tsm1.split('\t')
    uni_scores[count, 0] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_srs_tscore.values[0]
    uni_scores[count, 1] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_srs_rawscore.values[0]
    uni_scores[count, 2] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_srs_rawscore_self.values[0]
    uni_scores[count, 3] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_srs_tscore_combined.values[0]
    uni_scores[count, 4] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_srs_rawscore_combined.values[0]
    uni_scores[count, 5] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_ssp_total.values[0]
    uni_scores[count, 6] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_rbs_total.values[0]
    uni_scores[count, 7] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_adi_social_total.values[0]
    uni_scores[count, 8] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_adi_communication_total.values[0]
    uni_scores[count, 9] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_adi_rrb_total.values[0]
    uni_scores[count, 10] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_css_total_all.values[0]
    uni_scores[count, 11] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_sa_css_all.values[0]
    uni_scores[count, 12] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_rrb_css_all.values[0]
    uni_scores[count, 13] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_sdq_total_difficulties_p.values[0]
    uni_scores[count, 14] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_sdq_emotional_p.values[0]
    uni_scores[count, 15] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_sdq_conduct_p.values[0]
    uni_scores[count, 16] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_sdq_hyperactivity_p.values[0]
    uni_scores[count, 17] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_sdq_prosocial_p.values[0]
    uni_scores[count, 18] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_vabsdscoresc_dss.values[0]
    uni_scores[count, 19] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_vabsdscoresd_dss.values[0]
    uni_scores[count, 20] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_vabsdscoress_dss.values[0]
    uni_scores[count, 21] = clinical_info.loc[clinical_info.subjects == sbj_id].t1_vabsabcabc_standard.values[
        0]

[p_corrected, _, v_dl_ids] = ustats.run_controlling(X, ids, ages, sexes, sites, fsiqs, uni_scores[:,19], no_order)

# export csv for heatmap 

#plot scatterplot
scores = [19]
tsm_coefs = [[5,11]]
scores_names = ["Vineland-II Daily Living"]
tsm_coefs_names = [["$z^2$", "$z^4$"]]

for i, value in enumerate(scores):
    ids_remain, corrs, pvalues = su.run_univariate_correlation(X, uni_scores[:, value], ids, ages, sexes)
    [reject, pvals_corrected, _, alphacBonf] = multi.multipletests(pvalues, 0.05, 'holm')
    for ii, value_coef in enumerate(tsm_coefs[i]):
        [f, ax, slope, intercept, p_value] = su.plot_scatter(X, group, uni_scores[:, value], value_coef, corrs[value_coef],
                                                             pvals_corrected[value_coef], "lightslategrey", \
                                                             scores_names[i], tsm_coefs_names[i][ii],
                                                             scores_names[i])
        slopes[ii] = slope; intercepts[ii] = intercept
        f.tight_layout()
        f.savefig(defs.CONGRADS_OUTPUT_HARIRI + 'outputs/association_figs/w1/' + hms + '_' + scores_names[i] + '_' + tsm_coefs_names[i][ii] + '.png',
                  bbox_inches='tight', dpi=1000)
# use generate_reconstructions_y to plot the reconstructions of the connectopies
# at the speficied points across the vineland scale (figure 2)
