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

# NOTE: When writing tests, ensure there are no splits ('splitFactor' column
# always == 1) in any assets over the chosen simulation periods!

def _use_unadjusted(self):
    '''
    Use 'open' and 'close' columns in price dataFrames instead of 'adjClose' and
    'adjOpen'. The latter two are backwards-adjusted over time for dividends,
    splits, and so on, which makes them inappropriate for tests where we seek
    consistent results over time. **Keep in mind that this means these tests
    don't count as accurate backtests, just as examinations of each Strategy's
    inner workings.**

    Assigned to Strategy instances as a new method through MethodType.
    '''
    for _, val in self.assets.items():
        val['df'].rename(columns={'adjOpen': 'OG_adjO', 'adjClose': 'OG_adjC',
                                  'open': 'adjOpen', 'close': 'adjClose'},
                         inplace=True)

def _compare_figs(ax, img_root):
    '''
    Compare axes from plot_results() or plot_assets() in the test function's
    Strategy to those from a corresponding reference images.
    '''
    ref_img = f"{img_root}_ref.png"
    test_img = f"{img_root}_test.png"

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
    pf1.add_ticker('PEP', 1/3)
    pf1.add_ticker('SPY', 1, label='benchmark')

    # initialize Strategy and swap price columns
    bnh = BuyAndHoldStrategy(pf1,
                             start_date=pd.Timestamp(2000, 8, 1),
                             end_date=pd.Timestamp(2016, 1, 1),
                             cash=10000, reinvest_dividends=False,
                             tot_rb_freq=12, target_rb_day=0)

    bnh._use_unadjusted = MethodType(_use_unadjusted, bnh)
    bnh._use_unadjusted()

    # append another rebalance date to test the process
    nu_prices = {'MCD': 117.25, 'TGT': 71.84, 'PEP': 98.56, 'SPY': 200.49}
    bnh.append_date(pd.Timestamp(2016, 1, 4), nu_prices)

    # run simulation
    bnh.begin_time_loop()

    # compare portfolio values to expectations
    exp_pf_val = 43195.082500000004
    exp_bnch_val = 18642.541540100003
    test_pf_val = bnh.portfolio_value()
    test_bnch_val = bnh.portfolio_value(main_portfolio=False)

    assert test_pf_val == exp_pf_val, 'main portfolio value mismatch!'
    assert test_bnch_val == exp_bnch_val, 'benchmark portfolio value mismatch!'

    # compare portfolios' share counts to expectations
    exp_shares = {'MCD': 122.0, 'TGT': 200.0, 'PEP': 146.0, 'SPY': 92.0}
    tst_shares = {key : val['shares'] for key,val in bnh.assets.items()}
    assert tst_shares == exp_shares, 'portfolio share count mismatch!'

    # compare semi-log version of results plot to reference
    img_root = 'bnh'
    ax = bnh.plot_results(logy=True, return_plot=True, verbose=False)
    _compare_figs(ax, img_root)

    # compare semi-log version of ticker plot to reference
    img_root = 'bnh_tk'
    ax = bnh.plot_assets('MCD', 'TGT', 'PEP', 'SPY',
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

    sma._use_unadjusted = MethodType(_use_unadjusted, sma)
    sma._use_unadjusted()

    # run simulation
    sma.begin_time_loop()

    # compare portfolio values to expectations
    exp_pf_val = 15427.220752005629
    exp_bnch_val = 10753.115461066573
    test_pf_val = sma.portfolio_value()
    test_bnch_val = sma.portfolio_value(main_portfolio=False)

    assert test_pf_val == exp_pf_val, 'main portfolio value mismatch!'
    assert test_bnch_val == exp_bnch_val, 'benchmark portfolio value mismatch!'

    # compare portfolios' share counts to expectations
    exp_shares = {'SCHG': 38.84719129938834, 'SCHM': 20.764844216561272,
                  'EFG': 12.10906921207649, 'ACES': 12.177886217386732,
                  'BIV': 14.625274268414007, 'LQD': 3.042060286037833,
                  'FB': 0.0, 'TQQQ': 48.0, 'TLT': 0.0,
                  'SPY': 18.2881180227943, 'AGG': 35.52453216798228}
    tst_shares = {key : val['shares'] for key,val in sma.assets.items()}
    assert tst_shares == exp_shares, 'portfolio share count mismatch!'

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

    vlt._use_unadjusted = MethodType(_use_unadjusted, vlt)
    vlt._use_unadjusted()

    # run simulation
    vlt.begin_time_loop()

    # compare portfolio values to expectations
    exp_pf_val = 5942.462709999999
    exp_bnch_val = 6312.630000000001
    test_pf_val = vlt.portfolio_value()
    test_bnch_val = vlt.portfolio_value(main_portfolio=False)

    assert test_pf_val == exp_pf_val, 'main portfolio value mismatch!'
    assert test_bnch_val == exp_bnch_val, 'benchmark portfolio value mismatch!'

    # compare portfolios' share counts to expectations
    exp_shares = {'WMT': 16.0, 'SSO': 30.0, 'TLT': 2.0, 'GLD': 47.0}
    tst_shares = {key : val['shares'] for key,val in vlt.assets.items()}
    assert tst_shares == exp_shares, 'portfolio share count mismatch!'

    # compare results plot to reference
    img_root = 'vlt'
    ax = vlt.plot_results(return_plot=True, verbose=False)
    _compare_figs(ax, img_root)
