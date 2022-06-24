import numpy as np

def logloader(filepath, prefix='Average metric_0', ci=True):
    '''Load a running log file of a TFModel, and return repeat result (mean and 95CI) as a list

    Example
    -----
        r, ci = apdt.plt.logloader("slurm-190354.out")
        plt.errorbar(np.arange(len(r)), r, ci)
        plt.show()

    Parameter
    -----
        filepath: str
        prefix: str, default 'Average metric_0'
        sigma: bool, if return CI, which is considered to be at next line of prefix and startwith '95 confidence interval:  '

    Return
    -----
        list, numbers that following prefix
        list, numbers that maybe coressponding CI
    '''
    f = open(filepath, 'r')
    flag = 0
    result = []
    ci_result = []
    ci_prefix = '95 confidence interval:  '
    for line in f:
        if flag and ci and line.startswith(ci_prefix):
            ci_result.append(float(line[len(ci_prefix):].split(' ')[0]))
            flag = 0
        if line.startswith(prefix):
            result.append(float(line[len(prefix):].replace(' ','')))
            flag = 1
    
    if ci:
        return result, ci_result
    else:
        return result
