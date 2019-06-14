
import numpy as np

import matplotlib.pyplot as plt

from pipeline import psth


def unit_psth_ll(ipsi_hit, contra_hit, ipsi_err, contra_err):
    max_trial_off = 500

    plt_xmin = -3
    plt_xmax = 3
    plt_ymin = 0
    plt_ymax = None  # dynamic per unit

    plt_ymax = np.max([contra_hit['psth'][0],
                       ipsi_hit['psth'][0],
                       contra_err['psth'][0],
                       ipsi_err['psth'][0]])

    plt.figure()

    # raster plot
    ax = plt.subplot(411)
    plt.plot(contra_hit['raster'][0], contra_hit['raster'][1] + max_trial_off,
             'b.', markersize=1)
    plt.plot(ipsi_hit['raster'][0], ipsi_hit['raster'][1], 'r.', markersize=1)
    ax.set_axis_off()
    ax.set_xlim([plt_xmin, plt_xmax])
    ax.axvline(0, 0, 1, ls='--')
    ax.axvline(-1.2, 0, 1, ls='--')
    ax.axvline(-2.4, 0, 1, ls='--')

    # histogram of hits
    ax = plt.subplot(412)
    plt.plot(contra_hit['psth'][1][1:], contra_hit['psth'][0], 'b')
    plt.plot(ipsi_hit['psth'][1][1:], ipsi_hit['psth'][0], 'r')

    plt.ylabel('spikes/s')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([plt_xmin, plt_xmax])
    ax.set_ylim([plt_ymin, plt_ymax])
    ax.set_xticklabels([])
    ax.axvline(0, 0, 1, ls='--')
    ax.axvline(-1.2, 0, 1, ls='--')
    ax.axvline(-2.4, 0, 1, ls='--')
    plt.title('Correct trials')

    # histogram of errors
    ax = plt.subplot(413)
    plt.plot(contra_err['psth'][1][1:], contra_err['psth'][0], 'b')
    plt.plot(ipsi_err['psth'][1][1:], ipsi_err['psth'][0], 'r')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([plt_xmin, plt_xmax])
    ax.set_ylim([plt_ymin, plt_ymax])
    ax.axvline(0, 0, 1, ls='--')
    ax.axvline(-1.2, 0, 1, ls='--')
    ax.axvline(-2.4, 0, 1, ls='--')

    plt.title('Error trials')
    plt.xlabel('Time to go cue (s)')
    plt.show()


def unit_psth(unit_key):

    ipsi_hit_cond_key = (
        psth.TrialCondition() & {'condition_desc':
                                 'audio delay ipsi hit - nostim'}
    ).fetch1('KEY')

    contra_hit_cond_key = (
        psth.TrialCondition() & {'condition_desc':
                                 'audio delay contra hit - nostim'}
    ).fetch1('KEY')

    ipsi_err_cond_key = (
        psth.TrialCondition() & {'condition_desc':
                                 'audio delay ipsi error - nostim'}
    ).fetch1('KEY')

    contra_err_cond_key = (
        psth.TrialCondition() & {'condition_desc':
                                 'audio delay contra error - nostim'}
    ).fetch1('KEY')

    ipsi_hit_unit_psth = psth.UnitPsth.get(ipsi_hit_cond_key, unit_key)
    contra_hit_unit_psth = psth.UnitPsth.get(contra_hit_cond_key, unit_key)

    ipsi_err_unit_psth = psth.UnitPsth.get(ipsi_err_cond_key, unit_key)
    contra_err_unit_psth = psth.UnitPsth.get(contra_err_cond_key, unit_key)

    unit_psth_ll(ipsi_hit_unit_psth, contra_hit_unit_psth,
                 ipsi_err_unit_psth, contra_err_unit_psth)
