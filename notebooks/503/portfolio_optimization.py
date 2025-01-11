import numpy as np
import pandas as pd
import cvxpy as cp
import itertools
from scipy.optimize import minimize


"""
TODO:
    * add comments to code
    * increase number of iterations in opt
"""



def solve_cvxpy_problem(obj: cp.Minimize, constraints: list, solver=cp.ECOS) -> cp.Problem:
    """
    Solve problem in CVXPY given an objective and constraints
    """
    prob = cp.Problem(
        objective=obj,
        constraints=constraints,
    )
    
    prob.solve(solver=solver)
    assert prob.status == 'optimal'
    return prob

def solve_mean_var(rts: pd.DataFrame, target: float) -> np.array:
    """
    Solve mean-variance objective given a DataFrame of returns and a return target
    """
    mean = rts.mean().values
    cov = rts.cov().values
    n_assets = rts.shape[1]
    # define the vector we're solving
    w = cp.Variable(n_assets)
    
    constraints = [
        # sum of all weights is one
        cp.sum(w) == 1,
        # all weights non-negative
        w >= 0,
        # set the expected return
        w @ mean >= target,
    ]

    # minimize variance of portfolio
    obj = cp.Minimize(cp.quad_form(w, cov))
    prob = solve_cvxpy_problem(obj, constraints)
    return np.round(w.value, 6)

def solve_mean_mad(rts: pd.DataFrame, target: float) -> np.array:
    """
    Solve mean-MAD objective given a DataFrame of returns and a return target
    """    
    mean = rts.mean().values
    cov = rts.cov().values
    n_assets = rts.shape[1]    
    w = cp.Variable(n_assets)
    
    constraints = [
        # sum of all weights is one
        cp.sum(w) == 1,
        # all weights non-negative
        w >= 0,
        # set the expected return
        w @ mean >= target,
    ]
    
    obj = cp.Minimize(cp.sum(cp.abs((rts.values @ w) - (mean @ w))))
    prob = solve_cvxpy_problem(obj, constraints)
    return np.round(w.value, 6)

def var(xs, alpha):
    """
    Calculate VaR for a pandas Series
    """
    return np.percentile(xs, alpha, method='interpolated_inverted_cdf')

def cvar(xs, alpha):
    """
    Calcualte CVaR for a pandas Series
    """
    return xs[xs < var(xs, alpha)].mean()
    
def solve_mean_cvar(rts: pd.DataFrame, target: float, alpha: int | float) -> np.array:
    """
    Solve mean-MAD objective given a DataFrame of returns and a return target, and CVaR alpha, where alpha is given in npercent
    """      
    def objective(weights, pars):
        alpha = pars[0]
        portfolio_rts = (rts @ weights)
        return cvar(portfolio_rts, alpha) * -1
    
    # function to be used for total weight constraint
    def total_constraint(x, total_weight):
        return np.sum(x) - total_weight

    # function ot be used for targe return constraint
    def target_return_constraint(x, mean, target_return):
        return (x @ mean) - target_return

    mean = rts.mean().values
    cov = rts.cov().values
    n_assets = rts.shape[1]    
    
    # Initial guess for the weights
    w0 = np.ones(n_assets) / n_assets
    
    # Define bounds for the weights (weights should be between 0 and 1)
    bounds = [(0, 1) for _ in range(n_assets)]
    
    cons = (
        # sum of weights = 1
        {'type': 'eq', 'fun': total_constraint, 'args': [1]},
        # target return
        {'type': 'eq', 'fun': target_return_constraint, 'args': [mean, target]},
    )
    
    # Minimize the CVaR
    result = minimize(
        objective, 
        w0, 
        constraints=cons, 
        args=[alpha], 
        bounds=bounds
    )
    
    # Display results
    return np.round(result.x, 6)


def run_opt_partition(rts: pd.DataFrame, target: float, freq: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run mean-var, mean-MAD, and mean-CVaR optimization for a given set of returns and portfolio expected return target
    Output the resulting weights and basic statistics about the portfolio.
    """
    tgt = target * (freq / 252)
    sheet_name = f'days_{freq}'
    sheet_name = f'days_{freq:03d}'
    # run the different optimizations
    w_mv = solve_mean_var(rts, tgt)
    w_mad = solve_mean_mad(rts, tgt)
    w_cvar = solve_mean_cvar(rts, tgt, alpha=10)
    
    methods = ['Mean-Var', 'Mean-MAD', 'Mean-CVaR']
    ix = pd.MultiIndex.from_product(
        [[sheet_name], [str.format('{:,.1%}', target)], methods],
        names=['Frequency', 'Return Target', 'Method'])
    # construct a frame of weights for all opt methods
    wts = pd.DataFrame(
        index=list(rts.columns),
        columns=ix,
        data=np.array([w_mv,  w_mad, w_cvar]).T)
    # calculate portfolio returns
    rts_p = pd.DataFrame(
        index=list(rts.index),
        columns=methods,
        data=np.array([rts.values @ w_mv,  rts.values @ w_mad, rts.values @ w_cvar]).T)

    # calc stats before returning
    
    stats = pd.DataFrame(
        columns=methods,
    )
    stats.loc['mean', :] = rts_p.mean() * (252 / freq)
    stats.loc['stdev', :] = rts_p.std() * np.sqrt(252 / freq)
    stats.loc['mad', :] = np.sum(np.abs((rts.values @ wts.values) - (rts.mean().values @ wts.values)), axis=0) / rts_p.shape[0] * np.sqrt(252 / freq)
    stats.loc['var', :] = rts_p.apply(var, axis=0, alpha=10) * -1# * (252 / freq)    
    stats.loc['cvar', :] = rts_p.apply(cvar, axis=0, alpha=10) * -1# * (252 / freq)
    stats.columns = ix
    
    return (wts, stats)
    

def run_opts(asset_returns: dict, portfolio_return_targets: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point of the code. Run a series of optimizations given:
        * dictionary of DataFrames, each frame containing returns at different frequencies
        * list of portfolio return targets
    For each frequency/return target, run a mean-variance, mean-MAD, and mean-CVaR optimization. For each optimization
        we are assuming fully funded non-negative weights.
    """
    jobs = []
    wts_all = []
    stats_all = []
    for params in list(itertools.product(list(asset_returns.keys()), portfolio_return_targets)):
        # 
        sheet_name = params[0]
        freq = int(sheet_name.split('_')[-1])
        tgt = params[1]
        rts = asset_returns[sheet_name]
        rts['Date'] = pd.to_datetime(rts.Date).dt.date
        rts = rts.set_index('Date')
        wts_xs, stats_xs = run_opt_partition(rts, tgt, freq)
        wts_all.append(wts_xs)
        stats_all.append(stats_xs)

    wts = pd.concat(wts_all, axis=1)
    stats = pd.concat(stats_all, axis=1)
    
    return wts, stats


if __name__ == '__main__':
    
    all_data = pd.read_excel('Project1Data.xlsx', sheet_name=None)
    portfolio_return_targets = [0.02, 0.04, 0.06]
    
    wts, stats = opt.run_opts(all_data, portfolio_return_targets)    
