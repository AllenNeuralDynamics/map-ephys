'''
Descriptive analysis for the foraing task

1. Win-stay-lose-shift probabilities
2. Logistic regression on choice and reward history
   Use the model in Hattori 2019 https://www.sciencedirect.com/science/article/pii/S0092867419304465?via%3Dihub
            logit (p_R) ~ Rewarded choice + Unrewarded choice + Choice + Bias
  
Assumed format:
    choice = np.array([0, 1, 1, 0, ...])  # 0 = L, 1 = R
    reward = np.array([0, 0, 0, 1, ...])  # 0 = Unrew, 1 = Reward

Han Hou, Feb 2023
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import statsmodels.api as sm
import seaborn as sns


def win_stay_lose_shift(choice, reward, selected_trial_idx=None):
    '''
    Compute p(stay|win), p(shift|lose), and separately for two sides, i.e., p(stay|win in R), etc.
    
    choice = np.array([0, 1, 1, 0, ...])  # 0 = L, 1 = R
    reward = np.array([0, 0, 0, 1, ...])  # 0 = Unrew, 1 = Reward
    
    selected_trial_idx = np.array([selected zero-based trial idx]): 
            if None, use all trials; 
            else, only look at selected trials, but using the full history!
            e.g., p (stay at the selected trials | win at the previous trials of the selected trials) 
            therefore, the minimum idx of selected_trials is 1 (the second trial)
    
    ---
    return: dict{'p_stay_win', 'p_stay_win_CI', ...}
    '''
    
    stays = np.diff(choice) == 0  # stays[0] --> stay at the second trial (idx = 1)
    switches = np.diff(choice) != 0
    wins = reward[:-1] == 1
    loses = reward[:-1] == 0
    Ls = choice[:-1] == 0
    Rs = choice[:-1] == 1
    
    trial_mask = np.full(stays.shape, False)
    if selected_trial_idx is None:
        trial_mask[:] = True   # All trial used
    else:
        selected_trial_idx = selected_trial_idx[(selected_trial_idx >= 1) & (selected_trial_idx <= len(choice) - 1)]
        # assert not any((selected_trial_idx < 1) | (selected_trial_idx > len(choice) - 1)), "Wrong range for selected_trials!"
        trial_mask[selected_trial_idx - 1] = True
            
    p_wsls = {}
    p_lookup = {'p_stay_win':    (stays & wins & trial_mask, wins & trial_mask),   # 'p(y|x)': (y * x, x)
                'p_stay_win_L':  (stays & wins & Ls & trial_mask, wins & Ls & trial_mask),
                'p_stay_win_R':  (stays & wins & Rs & trial_mask, wins & Rs & trial_mask),
                'p_switch_lose': (switches & loses & trial_mask, loses & trial_mask),
                'p_switch_lose_L': (switches & loses & Ls & trial_mask, loses & Ls & trial_mask),
                'p_switch_lose_R': (switches & loses & Rs & trial_mask, loses & Rs & trial_mask),
                }

    for name, (k, n) in p_lookup.items():
        p_wsls[name], p_wsls[name + '_CI'] = _binomial(np.sum(k), np.sum(n))
        
    return p_wsls


def _binomial(k, n):
    '''
    Get p and its confidence interval
    '''
    p = k / n
    return p, 1.96 * np.sqrt(p * (1 - p) / n)


def prepare_logistic(choice, reward, trials_back=15, selected_trial_idx=None, **kwargs):
    '''    
    Assuming format:
    choice = np.array([0, 1, 1, 0, ...])  # 0 = L, 1 = R
    reward = np.array([0, 0, 0, 1, ...])  # 0 = Unrew, 1 = Reward
    trials_back: number of trials back into history
    selected_trial_idx = np.array([selected zero-based trial idx]): 
        if None, use all trials; 
        else, only look at selected trials, but using the full history!
        e.g., p (stay at the selected trials | win at the previous trials of the selected trials) 
        therefore, the minimum idx of selected_trials is 1 (the second trial)
    ---
    return: data, Y
    '''
    n_trials = len(choice)
    trials_back = 20
    data = []

    # Encoding data
    RewC, UnrC, C = np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials)
    RewC[(choice == 0) & (reward == 1)] = -1   # L rew = -1, R rew = 1, others = 0
    RewC[(choice == 1) & (reward == 1)] = 1
    UnrC[(choice == 0) & (reward == 0)] = -1    # L unrew = -1, R unrew = 1, others = 0
    UnrC[(choice == 1) & (reward == 0)] = 1
    C[choice == 0] = -1
    C[choice == 1] = 1

    # Select trials
    if selected_trial_idx is None:
        trials = range(trials_back, n_trials)
    else:
        trials = np.intersect1d(selected_trial_idx, range(trials_back, n_trials))
        
    for trial in trials:
        data.append(np.hstack([RewC[trial - trials_back : trial],
                            UnrC[trial - trials_back : trial], 
                            C[trial - trials_back : trial]]))
    data = np.array(data)
    
    Y = C[trials]  # Use -1/1 or 0/1?
    
    return data, Y


def logistic_regression(data, Y, solver='liblinear', penalty='l2', C=1, test_size=0.10, **kwargs):
    '''
    Run one logistic regression fit
    (Reward trials + Unreward trials + Choice + bias)
    Han 20230208
    '''
    trials_back = int(data.shape[1] / 3)
    
    # Do training
    # x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=test_size)
    logistic_reg = LogisticRegression(solver=solver, fit_intercept=True, penalty=penalty, C=C)

    # if sum(Y == 1) == 1 or sum(Y == -1) == 1:
    #     logistic_reg_cv.coef_ = np.zeros((1, data.shape[1]))
    #     logistic_reg_cv.intercept_ = 10 * np.sign(np.median(Y))   # If all left, set bias = 10 and other parameters 0
    #     logistic_reg_cv.C_ = np.nan
    # else:
    logistic_reg.fit(data, Y)

    output = np.concatenate([logistic_reg.coef_[0], logistic_reg.intercept_])
    
    (logistic_reg.b_RewC, 
    logistic_reg.b_UnrC, 
    logistic_reg.b_C, 
    logistic_reg.bias) = decode_betas(output)
    
    return output, logistic_reg


def logistic_regression_CV(data, Y, Cs=10, cv=10, solver='liblinear', penalty='l2', n_jobs=-1):
    '''
    logistic regression with cross validation
    1. Use cv-fold cross validation to determine best penalty C
    2. Using the best C, refit the model with cv-fold again
    3. Report the mean and CI (1.96 * std) of fitted parameters in logistic_reg_refit
    
    Cs: number of Cs to grid search
    cv: number of folds
    
    -----
    return: logistic_reg_cv, logistic_reg_refit
    
    Han 20230208
    '''

    # Do cross validation, try different Cs
    logistic_reg_cv = LogisticRegressionCV(solver=solver, fit_intercept=True, penalty=penalty, Cs=Cs, cv=cv, n_jobs=n_jobs)
    
    # if sum(Y == 1) == 1 or sum(Y == -1) == 1:
    #     logistic_reg_cv.coef_ = np.zeros((1, data.shape[1]))
    #     logistic_reg_cv.intercept_ = 10 * np.sign(np.median(Y))   # If all left, set bias = 10 and other parameters 0
    #     logistic_reg_cv.C_ = np.nan
    # else:
    logistic_reg_cv.fit(data, Y)

    return logistic_reg_cv


def bootstrap(func, data, Y, n_bootstrap=1000, n_samplesize=None, **kwargs):
    # Generate bootstrap samples
    indices = np.random.choice(range(Y.shape[0]), size=(n_bootstrap, Y.shape[0] if n_samplesize is None else n_samplesize), replace=True)   # Could do subsampling
    bootstrap_Y = [Y[index] for index in indices]
    bootstrap_data = [data[index, :] for index in indices]
    
    # Fit the logistic regression model to each bootstrap sample
    outputs = np.array([func(data, Y, **kwargs)[0] for data, Y in zip(bootstrap_data, bootstrap_Y)])
    
    # Get bootstrap mean, std, and CI
    bs = {'raw': outputs,
          'mean': np.mean(outputs, axis=0),
          'std': np.std(outputs, axis=0),
          'CI_lower': np.percentile(outputs, 2.5, axis=0),
          'CI_upper': np.percentile(outputs, 97.5, axis=0)}
    
    return bs
    
    
def decode_betas(coef):
    # Decode fitted betas
    coef = np.atleast_2d(coef)
    trials_back = int((coef.shape[1] - 1) / 3)  # Hard-coded
    b_RewC = coef[:, trials_back - 1::-1]
    b_UnrC = coef[:, 2 * trials_back - 1: trials_back - 1:-1]
    b_C = coef[:, 3 * trials_back - 1:2 * trials_back - 1:-1]
    bias = coef[:, -1:]
    return b_RewC, b_UnrC, b_C, bias


def logistic_regression_bootstrap(data, Y, n_bootstrap=1000, n_samplesize=None, **kwargs):
    '''
    1. use cross-validataion to determine the best L2 penality parameter, C
    2. use bootstrap to determine the CI and std
    '''
    
    # Cross validation
    logistic_reg = logistic_regression_CV(data, Y, **kwargs)
    best_C = logistic_reg.C_
    para_mean = np.hstack([logistic_reg.coef_[0], logistic_reg.intercept_])
    
    (logistic_reg.b_RewC, 
     logistic_reg.b_UnrC, 
     logistic_reg.b_C, 
     logistic_reg.bias) = decode_betas(para_mean)
    
    # Bootstrap
    if n_bootstrap > 0:
        bs = bootstrap(logistic_regression, data, Y, n_bootstrap=n_bootstrap, n_samplesize=n_samplesize, C=best_C[0], **kwargs)
        
        logistic_reg.coefs_bootstrap = bs
        (logistic_reg.b_RewC_CI, 
        logistic_reg.b_UnrC_CI, 
        logistic_reg.b_C_CI, 
        logistic_reg.bias_CI) = decode_betas(np.vstack([bs['CI_lower'], bs['CI_upper']]))

        # # Override with bootstrap mean
        # (logistic_reg.b_RewC, 
        # logistic_reg.b_UnrC, 
        # logistic_reg.b_C, 
        # logistic_reg.bias) = decode_betas(np.vstack([bs['mean'], bs['mean']]))
    
    return logistic_reg



# --- Linear regression of z-scored RT ----
def prepare_linear_reg_RT(choice, reward, reaction_time, iti, trials_back=10, selected_trial_idx=None):
    n_trials = len(choice)
    data = []

    # Select trials
    if selected_trial_idx is None:
        trials = range(trials_back, n_trials)
    else:
        trials = np.intersect1d(selected_trial_idx, range(trials_back, n_trials))

    # Design matrix
    for trial in trials:
        data.append(np.hstack([
                            (iti[trial] - np.mean(iti)) / np.std(iti),      # normalized iti means before the trial here
                            #reaction_time[trial - 1],   # last RT
                            choice[trial],   # current choice (accounting for licking bias)
                            trial / n_trials,   # normalized trial number
                            reward[trial - trials_back : trial][::-1],  # reward history
                            ]))
    X = np.array(data)
    Y = reaction_time[trials]
    Y = (Y - np.nanmean(reaction_time)) / np.nanstd(reaction_time)  # z-scored using all reaction times of this session, not just selected trials
    
    # remove nans in Y, just in case
    valid_Y = ~np.isnan(Y)
    Y = Y[valid_Y]
    X = X[valid_Y, :]
    
    x_name = ['constant', 'previous_iti', 'this_choice', 'trial_number', 
              *[f'reward (-{x})' for x in range(1, trials_back + 1)]]
    
    return X, Y, x_name


def linear_regression(X, Y, x_name=None):
    model = sm.OLS(Y, sm.add_constant(X), xname=x_name)
    linear_reg = model.fit()
    linear_reg.x_name = x_name
    return linear_reg


# ----- Plotting functions -----
def plot_logistic_regression(logistic_reg, ax=None, ls='-o'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        
    # return 
    if_CV = hasattr(logistic_reg, 'b_RewC_CI') # If cross-validated
    x = np.arange(1, logistic_reg.b_RewC.shape[1] + 1)
    plot_spec = {'b_RewC': 'g', 'b_UnrC': 'r', 'b_C': 'b', 'bias': 'k'}    

    for name, col in plot_spec.items():
        mean = getattr(logistic_reg, name)
        ax.plot(x if name != 'bias' else 1, np.atleast_2d(mean)[0, :], ls + col, label=name + ' $\pm$ CI')

        if if_CV:  # From cross validation
            CI = np.atleast_2d(getattr(logistic_reg, name + '_CI'))
            ax.fill_between(x=x if name != 'bias' else [1], 
                            y1=CI[0, :], 
                            y2=CI[1, :], 
                            color=col,
                            alpha=0.3)
        
    if if_CV and hasattr(logistic_reg, "scores_"):
        score_mean = np.mean(logistic_reg.scores_[1.0])
        score_std = np.std(logistic_reg.scores_[1.0])
        if hasattr(logistic_reg, 'cv'):
            ax.set(title=f'{logistic_reg.cv}-fold CV, score $\pm$ std = {score_mean:.3g} $\pm$ {score_std:.2g}\n'
                    f'best C = {logistic_reg.C_[0]:.3g}')
    else:
        pass
        # ax.set(title=f'train: {logistic_reg.train_score:.3g}, test: {logistic_reg.test_score:.3g}')
    
    ax.legend()
    ax.set(xlabel='Past trials', ylabel='Logistic regression coeffs')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    
    return ax


def plot_logistic_compare(logistic_to_compare, 
                          past_trials_to_plot = [1, 2, 3, 4],
                          labels=['ctrl', 'photostim', 'photostim_next'], 
                          edgecolors=['None', 'deepskyblue', 'skyblue'],
                          plot_spec = {'b_RewC': 'g', 'b_UnrC': 'r', 'b_C': 'b', 'bias': 'k'},
                          ax_all=None):
    
    '''
    Compare logistic regressions. Columns for betas, Rows for past trials
    '''
    
    if ax_all is None:   # ax_all is only one axis
        fig, ax_all = plt.subplots(1, 1, figsize=(10, 3 * len(past_trials_to_plot)), layout="constrained")

    # add subaxis in ax_all
    gs = ax_all._subplotspec.subgridspec(len(past_trials_to_plot), len(plot_spec))
    axes = []
        
    for i, past_trial in enumerate(past_trials_to_plot):
        for j, (name, col) in enumerate(plot_spec.items()):
            # ax = axes[i, j]
            ax = ax_all.get_figure().add_subplot(gs[i, j])
            axes.append(ax)
            
            if name == 'bias' and i > 0: 
                ax.set_ylim(0, 0)
                ax.remove()
                continue

            for k, logistic in enumerate(logistic_to_compare):
                mean = getattr(logistic, f'{name}')[0, past_trial - 1]
                yerrs = np.abs(getattr(logistic, f'{name}_CI')[:, past_trial - 1:past_trial] - mean)
                ax.plot(k, mean, marker='o', color=col, markersize=10, markeredgecolor=edgecolors[k], markeredgewidth=2)
                ax.errorbar(x=k, 
                            y=mean,
                            yerr=yerrs,
                            fmt='o', color=col, markeredgecolor=edgecolors[k],
                            ecolor=col, lw=2, capsize=5, capthick=2)

                if i == len(past_trials_to_plot): plt.xticks([0, 1, 2], labels=labels, rotation=45)

            ax.set(xlim=[-0.5, 0.5 + k])
            ax.axhline(y=0, linestyle='--', c='k', lw=1)
            ax.spines[['right', 'top']].set_visible(False)

            if i == 0:
                ax.set_title(name)

            if i == len(plot_spec) - 1: 
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
            else:
                ax.set_xticks([])

            if j == 0: 
                ax.set_ylabel(f'past_trial = {past_trial}')
            else:
                ax.set_yticklabels([])    


    ylim_min = min([ax.get_ylim()[0] for ax in axes])
    ylim_max = max([ax.get_ylim()[1] for ax in axes])
    for ax in axes:
        ax.set_ylim(ylim_min, ylim_max)
        
    ax_all.remove()
    
    return axes



def plot_linear_regression_RT(regs, ax=None, 
                              cols=['k', 'royalblue', 'lightskyblue'], 
                              labels=['control', 'photostim', 'photostim+1'],
                              offset=0.2):
    '''
    plot list of linear regressions
    '''
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        
    gs = ax._subplotspec.subgridspec(1, 2, width_ratios=[1, 3], wspace=0.2)
    ax_others = ax.get_figure().add_subplot(gs[0, 0])
    ax_reward = ax.get_figure().add_subplot(gs[0, 1])
    
    if not isinstance(regs, list):
        regs = [regs]   
        
    for i_reg, (reg, col) in enumerate(zip(regs, cols)):
    
        paras = {name: (value, ci, p) for name, value, ci, p in zip(reg.x_name, 
                                                                    reg.params,
                                                                    reg.params - reg.conf_int(alpha=0.05)[:, 0],
                                                                    reg.pvalues)}
        
        names = ['previous_iti', 'this_choice', 'trial_number', 'constant']
        xx = np.arange(len(names))
        yy = [paras[name][0] for name in names]
        yerr = [paras[name][1] for name in names]
        sigs = [paras[name][2] < 0.05 for name in names]
        ax_others.errorbar(x=xx + i_reg * offset,
                           y=yy, yerr=yerr,
                        ls='none', color=col, capsize=5, markeredgewidth=1,
                        )
        ax_others.scatter(x=xx + i_reg * offset, 
                          y=yy,
                        facecolors=[col if sig else 'none' for sig in sigs],
                        marker='o', edgecolors=col, linewidth=1,
                        )
        ax_others.set_xlim(-1, 4)
        
        ax_others.set_xticks(range(len(names)))
        ax_others.set_xticklabels(names, rotation=45, ha='right')
        ax_others.axhline(y=0, color='k', linestyle=':', linewidth=1)
        ax_others.set(ylabel='fitted $\\beta\pm$95% CI ')
    
        xx = np.arange(1, len(paras) - 4 + 1)
        yy = [paras[f'reward (-{x})'][0] for x in xx]
        yerr = [paras[f'reward (-{x})'][1] for x in xx]
        sigs = [paras[f'reward (-{x})'][2] < 0.05 for x in xx]
        ax_reward.errorbar(x=xx + i_reg * offset, 
                           y=yy, yerr=yerr,
                        color=col, capsize=5, markeredgewidth=1
                        )
        ax_reward.scatter(x=xx + i_reg * offset,
                          y=yy,
                        facecolors=[col if sig else 'none' for sig in sigs],
                        marker='o', edgecolors=col, linewidth=1,
                        label=f'{labels[i_reg]}, n = {int(reg.nobs)}',
                        )
        
        ax_reward.legend(loc='upper right')

        ax_reward.set(xlabel='Reward of past trials')
        ax_reward.axhline(y=0, color='k', linestyle=':', linewidth=1)
        ax_reward.set(xticks=[1, 5, 10])
    
    ax_reward.invert_yaxis()
    
    sns.despine(trim=True)
    ax.get_figure().suptitle('Linear regression on RT')

    ax.remove()
    
    return ax



def plot_session_linear_reg_RT(choice, reward, reaction_time, iti, 
                               photostim_idx=None, ax=None):
    
    if photostim_idx is None:
        # Fit models for control, photostim, photostim_next
        data, Y, x_name = prepare_linear_reg_RT(choice, reward, reaction_time, iti)
        linear_reg = linear_regression(data, Y, x_name)
        plot_linear_regression_RT([linear_reg], ax=ax)
        
        return [linear_reg]

    else:

        ctrl_idx = np.setdiff1d(np.arange(len(choice)), photostim_idx)

        # Fit models for control, photostim, photostim_next
        data_ctrl, Y_ctrl, x_name = prepare_linear_reg_RT(choice, reward, reaction_time, iti, selected_trial_idx=ctrl_idx)
        linear_reg = linear_regression(data_ctrl, Y_ctrl, x_name)

        data_photostim0, Y_photostim0, _ = prepare_linear_reg_RT(choice, reward, reaction_time, iti, selected_trial_idx=photostim_idx)
        linear_reg_photostim0 = linear_regression(data_photostim0, Y_photostim0, x_name)

        data_photostim1, Y_photostim1, _ = prepare_linear_reg_RT(choice, reward, reaction_time, iti, selected_trial_idx=photostim_idx + 1)
        linear_reg_photostim1 = linear_regression(data_photostim1, Y_photostim1, x_name)

        plot_linear_regression_RT([linear_reg, linear_reg_photostim0, linear_reg_photostim1], ax=ax)
    
        return [linear_reg, linear_reg_photostim0, linear_reg_photostim1]



def plot_wsls(p_wslss, ax=None, edgecolors=None, labels=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    xlabel = []
    
    if not isinstance(p_wslss, list):
        p_wslss = [p_wslss]
        
    for i, name in enumerate(['stay_win', 'switch_lose']):
        for n, p_wsls in enumerate(p_wslss):       
            ax.bar(i * len(p_wslss) * 3 + n, 
                   p_wsls[f'p_{name}'], 
                   yerr=p_wsls[f'p_{name}_CI'],
                   color='k', 
                   label='all $\pm$ CI' if labels is None else f'all $\pm$ CI {labels[n]}',
                   edgecolor=edgecolors[n] if edgecolors is not None else None,
                   linewidth=5,
                   error_kw=dict(ecolor='k' , lw=2, capsize=5, capthick=2))
            for j, (side, col) in enumerate(zip(('L', 'R'), ('r', 'b'))):
                ax.bar((i * 3 + j + 1) * len(p_wslss) + n, 
                       p_wsls[f'p_{name}_{side}'], 
                       yerr=p_wsls[f'p_{name}_{side}_CI'],
                       color=col, 
                       label=f'{side} $\pm$ CI',
                       edgecolor=edgecolors[n] if edgecolors is not None else None,
                       linewidth=5,
                       error_kw=dict(ecolor=col , lw=2, capsize=5, capthick=2))
    
    ax.set(xticks=[len(p_wslss) * 3 / 2, len(p_wslss) * 3 + len(p_wslss) * 3 / 2], 
           xticklabels=['p(stay | win)', 'p(shift | lose)'])
    ax.xaxis.set_ticks_position('none') 
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:3], l[:3])
    ax.axhline(y=1, color='k', linestyle='--', linewidth=1)
    # ax.set(ylim=(0, 1))
    
    return ax
    

# --- Wrappers ---
def do_logistic_regression(choice, reward, **kwargs):
    data, Y = prepare_logistic(choice, reward, **kwargs)
    logistic_reg = logistic_regression_bootstrap(data, Y, **kwargs)
    return plot_logistic_regression(logistic_reg)

def do_wsls(choice, reward):
    p_wsls = win_stay_lose_shift(choice, reward)
    return plot_wsls(p_wsls)
