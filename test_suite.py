import numpy as np
import os
import pandas as pd
import pytest

from matplotlib.testing.compare import compare_images
from types import MethodType
from portfolio_maker import PortfolioMaker
from strategies import BuyAndHoldStrategy, SMAStrategy, VolTargetStrategy

# Each test function focuses on a Strategy and ensures that, after the time
# loop, the closing portfolio values (main + benchmark) and share counts equal
# their expected values. Individual functions may also test other things; these
# are mentioned in their docstrings.

def _rewind_prices(self):
    '''
    Takes a dataFrame from a symbol's 'df' key in the `assets` dictionary of a
    Strategy instance (e.g. bnh.assets['AAPL']['df']) and adjusts the
    dataFrame's 'adj' columns (close, high, low, open) to approximate how they
    would have looked on the Strategy instance's end date instead of on
    whichever date the instance was created.

    This is useful because, as seen in cron job results, Tiingo's own 'adj'
    column values vary over time as dividends and stock splits occur. To get
    consistent results for testing purposes, you must either use the unadjusted
    columns (ignoring dividends and splits) or figure out how to reproducibly
    adjust the price values to your own liking (this!).

    Assigned to Strategy instances as a new method through MethodType.

    NOTE: Past dates' 'adj' values won't exactly match the originals for
    Strategy instances with end dates on your present date. This is likely
    because the original 'close' column is rounded to only the hundredths
    place. It should only be a small discrepancy: for AAPL from 1993-12-27 to
    2021-02-10, the difference in 'adjClose' values is 3e-5 on the first day.

    NOTE: VALUES WILL NEED TO BE CHANGED AGAIN ONCE run_time_loop IS CORRECTED
    SUCH THAT IT DOESN'T DOUBLE COUNT DIVIDENDS.
    '''
    for _, val in self.assets.items():
        df = val['df'].copy()
        cols = ['close', 'high', 'low', 'open']

        # split factor for each date through cumulative product
        # (inspiration from https://stackoverflow.com/questions/62130566/)
        tot_splits = df['splitFactor'].cumprod().values
        tot_splits_cast = np.tile(tot_splits, (len(cols), 1)).T

        # get dates and amounts of dividends
        div_dts = df[df['divCash'] != 0].index
        div_amts = df.loc[div_dts, 'divCash'].values
        div_amts_cast = np.tile(div_amts, (len(cols), 1)).T

        # get indices of days that precede dividend ex-dates
        pre_div_inds = df.reset_index()[df.index.isin(div_dts)].index - 1

        # get dates and prices of these pre-dividend indices
        pre_div_dts = df.iloc[pre_div_inds].index
        pre_div_prices = df.iloc[pre_div_inds][cols]
        # (pre_div_closes, pre_div_highs,
        #  pre_div_lows, pre_div_opens) = df.iloc[pre_div_inds][cols].T.values

        # calculate dividend adjustments on close prices, then get
        # cumulative product of these adjustments going backward in time
        tot_div_adjs = ( (pre_div_prices - div_amts_cast)
                         / pre_div_prices)[::-1].cumprod()[::-1]

        # apply the dividend adjustments to previous prices
        divFactors = pd.DataFrame(index=df.index, columns=cols, data=1,
                                  dtype=np.float64)

        for i, dt in enumerate(pre_div_dts):
            if i == 0:
                divFactors.loc[:pre_div_dts[i]] = tot_div_adjs.iloc[i].T.values
            else:
                divFactors.loc[div_dts[i-1] : pre_div_dts[i],
                               :] = tot_div_adjs.iloc[i].T.values

        # rename Tiingo's adj columns
        adj_cols = ['adj' + c.title() for c in cols]
        old_adj_cols = ['OG_' + c.title() for c in adj_cols]
        renames = {adj: old for adj, old in zip(adj_cols, old_adj_cols)}

        val['df'].rename(columns=renames, inplace=True)

        # add the readjusted price colums to the original dataFrame
        val['df'][adj_cols] = tot_splits_cast * divFactors * df[cols]

def _compare_figs(ax, img_root):
    '''
    Compare axes from plot_results() or plot_assets() in the test function's
    Strategy to those from a corresponding reference images.
    '''
    ref_img = f"tests/{img_root}_ref.png"
    test_img = f"tests/{img_root}_test.png"

    ax.figure.savefig(test_img, facecolor='w', dpi=75)
    msg = compare_images(ref_img, test_img, tol=15)
    # allow leeway in tolerance for axis labels, legend, etc.

    if msg is not None:
        pytest.fail(msg, pytrace=False)
    else:
        os.remove(test_img)

def test_bnh():
    '''
    Deals with BuyAndHoldStrategy. In addition to common focuses, also tests
    output from HistoricalSimulator's append_date, plot_results, and
    plot_assets methods, using semi-log plots for the latter two. Uses a
    core-only portfolio by necessity.
    '''
    # build portfolio
    pf1 = PortfolioMaker(sat_frac=0)
    pf1.add_ticker('MCD', 1/3)
    pf1.add_ticker('TGT', 1/3)
    pf1.add_ticker('CPB', 1/3)
    pf1.add_ticker('SPY', 1, label='benchmark')

    # initialize Strategy and swap price columns
    bnh = BuyAndHoldStrategy(pf1,
                             start_date=pd.Timestamp(2000, 8, 1),
                             end_date=pd.Timestamp(2016, 1, 1),
                             cash=10000, reinvest_dividends=False,
                             tot_rb_freq=12, target_rb_day=0)

    bnh._rewind_prices = MethodType(_rewind_prices, bnh)
    bnh._rewind_prices()

    # append another rebalance date to test the process
    nu_prices = {'MCD': 117.25, 'TGT': 71.84, 'CPB': 43.83, 'SPY': 200.49}
    bnh.append_date(pd.Timestamp(2016, 1, 4), nu_prices)

    # run simulation
    bnh.begin_time_loop()

    # compare portfolio values to expectations
    exp_pf_val = 62600.62671775877
    exp_bnch_val = 26162.867776345214
    test_pf_val = bnh.portfolio_value()
    test_bnch_val = bnh.portfolio_value(main_portfolio=False)

    np.testing.assert_almost_equal(test_pf_val, exp_pf_val, decimal=4,
                                   err_msg=': main portfolio')
    np.testing.assert_almost_equal(test_bnch_val, exp_bnch_val, decimal=4,
                                   err_msg=': benchmark portfolio')

    # compare portfolios' share counts to expectations
    exp_shares = {'MCD': 177.0, 'TGT': 290.0, 'CPB': 476.0, 'SPY': 130.0}
    tst_shares = {key : val['shares'] for key,val in bnh.assets.items()}
    assert tst_shares == exp_shares, 'portfolio share count mismatch'

    # compare semi-log version of results plot to reference
    img_root = 'bnh'
    ax = bnh.plot_results(logy=True, return_plot=True, verbose=False)
    _compare_figs(ax, img_root)

    # compare semi-log version of ticker plot to reference
    img_root = 'bnh_tk'
    ax = bnh.plot_assets('MCD', 'TGT', 'CPB', 'SPY',
                         logy=True, return_plot=True, verbose=False)
    _compare_figs(ax, img_root)

def test_sma():
    '''
    Deals with SMAStrategy. In addition to common focuses, also tests
    HistoricalSimulator's plot_results method and how PortfolioMaker handles
    tickers with initial shares.

    '''
    # build portfolio (more complex this time)
    pf2 = PortfolioMaker(sat_frac=.4, relative_core_frac=True)
    pf2.add_ticker('SCHG', .48, label='core', shares=2.7)
    pf2.add_ticker('SCHM', .13, label='core', shares=2.6)
    pf2.add_ticker('EFG', .12, label='core', shares=12)
    pf2.add_ticker('ACES', .07, label='core')
    pf2.add_ticker('BIV', .15, label='core', shares=.445)
    pf2.add_ticker('LQD', .05, label='core', shares=5)
    pf2.add_ticker('FB', 0, label='core', shares=30)
    pf2.add_ticker('TQQQ', label='satellite', in_market=True)
    pf2.add_ticker('TLT', label='satellite', in_market=False)
    pf2.add_ticker('SPY', .6, label='benchmark', track=True)
    pf2.add_ticker('AGG', .4, label='benchmark')

    # initialize Strategy and swap price columns
    sma = SMAStrategy(pf2, window=100,
                      start_date=pd.Timestamp(2019, 12, 12),
                      end_date=pd.Timestamp(2020, 8, 13),
                      cash=1738.29, reinvest_dividends=True,
                      sat_rb_freq=365.25, tot_rb_freq=12, target_rb_day=8)

    sma._rewind_prices = MethodType(_rewind_prices, sma)
    sma._rewind_prices()

    # run simulation
    sma.begin_time_loop()

    # compare portfolio values to expectations
    exp_pf_val = 14038.487133948604
    exp_bnch_val = 10898.582343484206
    test_pf_val = sma.portfolio_value()
    test_bnch_val = sma.portfolio_value(main_portfolio=False)

    np.testing.assert_almost_equal(test_pf_val, exp_pf_val, decimal=4,
                                   err_msg=': main portfolio')
    np.testing.assert_almost_equal(test_bnch_val, exp_bnch_val, decimal=4,
                                   err_msg=': benchmark portfolio')

    # compare portfolios' share counts to expectations
    exp_shares = {'SCHG': 35.83601442282118, 'SCHM': 18.748962642069873,
                  'EFG': 11.109352277298585, 'ACES': 11.169197042734888,
                  'BIV': 12.618948407993443, 'LQD': 3.040265631432126,
                  'FB': 0.0, 'TQQQ': 43.0, 'TLT': 0.0,
                  'SPY': 19.300408108960365, 'AGG': 36.53405970562885}
    tst_shares = {key : val['shares'] for key,val in sma.assets.items()}

    for key in tst_shares.keys():
        np.testing.assert_almost_equal(tst_shares[key], exp_shares[key],
                                       decimal=4,
                                       err_msg=f": {key} share count")

    # compare results plot to reference
    img_root = 'sma'
    ax = sma.plot_results(return_plot=True, verbose=False)
    _compare_figs(ax, img_root)

def test_vlt():
    '''
    Deals with VolTargetStrategy. In addition to common focuses, also tests
    HistoricalSimulator's plot_results.

    NOTE: VolTargetStrategy has problems when reinvest_dividends=True because
    it can mostly only sell whole shares. When fractions build up, it thinks
    there's $$ to spend that actually can't be touched unless it's entirely
    liquidating an asset that has fractional shares. it could really use a
    method like "functional_portfolio_value" that multiplies share prices by
    the floor of the number of shares of each satellite asset.)
    '''
    # build portfolio
    pf3 = PortfolioMaker(sat_frac=.7, relative_core_frac=False)
    pf3.add_ticker('WMT', .3, label='core')
    pf3.add_ticker('SSO', label='satellite', in_market=True)
    pf3.add_ticker('TLT', label='satellite', in_market=False)
    pf3.add_ticker('GLD', 1, label='benchmark')

    # initialize Strategy and swap price columns
    vlt = VolTargetStrategy(pf3, window=30, vol_target=.15,
                            start_date=pd.Timestamp(2018, 7, 27),
                            end_date=pd.Timestamp(2019, 7, 31),
                            cash=5500, reinvest_dividends=False,
                            sat_rb_freq=12, tot_rb_freq=4, target_rb_day=-3)

    vlt._rewind_prices = MethodType(_rewind_prices, vlt)
    vlt._rewind_prices()

    # run simulation
    vlt.begin_time_loop()

    # compare portfolio values to expectations
    exp_pf_val = 6134.986341181822
    exp_bnch_val = 6312.630000000001
    test_pf_val = vlt.portfolio_value()
    test_bnch_val = vlt.portfolio_value(main_portfolio=False)

    np.testing.assert_almost_equal(test_pf_val, exp_pf_val, decimal=4,
                                   err_msg=': main portfolio value')
    np.testing.assert_almost_equal(test_bnch_val, exp_bnch_val, decimal=4,
                                   err_msg=': benchmark portfolio value')

    # compare portfolios' share counts to expectations
    exp_shares = {'WMT': 16.0, 'SSO': 31.0, 'TLT': 2.0, 'GLD': 47.0}
    tst_shares = {key : val['shares'] for key,val in vlt.assets.items()}
    assert tst_shares == exp_shares, 'portfolio share count mismatch'

    # compare results plot to reference
    img_root = 'vlt'
    ax = vlt.plot_results(return_plot=True, verbose=False)
    _compare_figs(ax, img_root)
