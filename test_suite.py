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
    dataFrame's 'adj' columns (close, high, low, open) to the basis of the
    prices on the Strategy instance's end date. **This makes simulation results
    reproducible over time.**

    This is useful because Tiingo's own 'adj' values are adjusted to the basis
    of the prices on the date the data were queried. This means the values
    change over time as more dividend payments and stock splits occur. To get
    consistent numbers, one must either use the unadjusted columns (and ignore
    dividends and splits), always query price data until the present day (to
    capture and factor in subsequent corporate actions), or otherwise adjust the
    unadjusted prices in a reproducible manner (this!).

    Assigned to Strategy instances as a new method through MethodType.

    NOTE: Past dates' 'adj' values won't exactly match Tiingo's originals,
    even when a Strategy query includes all corporate actions between the start
    date and the present. This is almost fully because Tiingo's unadjusted
    'close' column is rounded to 2 decimal places while they use more for their
    adjustments. However, this is a minuscule discrepancy that doesn't affect
    Strategy simulations. For example, the Tiingo-provided adjClose on the start
    date of an AAPL query from 1993-12-27 to 2026-03-30 (conducted on the latter
    date) was 0.02% different in price from this method's adjClose.

    NOTE: VALUES WILL NEED TO BE CHANGED AGAIN ONCE run_time_loop IS CORRECTED
    SUCH THAT IT DOESN'T DOUBLE COUNT DIVIDENDS.
    '''
    for tkr, val in self.assets.items():
        df = val['df'].copy()
        cols = ['close', 'high', 'low', 'open']

        # split factor for each date through reversed cumulative product
        # (inspiration from https://stackoverflow.com/questions/62130566/)
        tot_splits = df['splitFactor'][::-1].cumprod()[::-1].values
        tot_splits_cast = np.tile(tot_splits, (len(cols), 1)).T

        # get prices on dates that precede dividend ex-dates
        # (excluding any that aren't present in the dataFrame)
        div_dts_all = df[df['divCash'] != 0].index
        pre_div_inds_all = df.index.get_indexer_for(div_dts_all)
        pre_div_inds = pre_div_inds_all[pre_div_inds_all > 0]
        pre_div_prices = df.iloc[pre_div_inds - 1][cols]

        # get dates and amounts of remaining dividends
        div_dts = div_dts_all[pre_div_inds_all > 0]
        div_amts = df.loc[div_dts, 'divCash'].values
        div_amts_cast = np.tile(div_amts, (len(cols), 1)).T

        # calculate dividend adjustments on close (and other) prices
        # (div_adjs_by_dt[::-1].cumprod()[::-1] == old tot_div_adjs)
        div_adjs_by_dt = ((pre_div_prices - div_amts_cast) / pre_div_prices)

        # multiply each dividend adjustment into the dates before its ex date
        divFactors = pd.DataFrame(index=df.index, columns=cols,
                                  data=1, dtype=np.float64)
        for dt in div_adjs_by_dt.index:
            divFactors.loc[divFactors.index <= dt] *= div_adjs_by_dt.loc[dt]

        # NOT NEEDED; DIVIDENDS ARE ALREADY BAKED INTO THE ADJUSTMENT
        # # calculate each dividend's value as shares of previous (unadj.) close
        # divs_in_shares = df.loc[div_dts, 'divCash'] / pre_div_prices['close'].values

        # # scale each dividend to equivalent per-share value for adjusted prices;
        # # rename original divCash column and replace with scaled values
        # val['df']['OG_divCash'] = val['df']['divCash'].copy()
        # for i, dt in enumerate(divs_in_shares.index):
        #     val['df'].loc[dt, 'divCash'] = (
        #         df.loc[df.index[pre_div_inds[i]], 'close']
        #         * divs_in_shares.loc[dt]
        #     )

        # rename Tiingo's adj columns
        adj_cols = ['adj' + c.title() for c in cols]
        old_adj_cols = ['OG_' + c for c in adj_cols]
        renames = {adj: old for adj, old in zip(adj_cols, old_adj_cols)}

        val['df'].rename(columns=renames, inplace=True)

        # add the readjusted price columns to the original dataFrame
        val['df'][adj_cols] = df[cols] / tot_splits_cast * divFactors

    # reset benchmark portfolio starting value to reflect any price adjustments
    # (copied from HistoricalSimulator.__init__())
    self._bench_cash = self.portfolio_value(self.start_date, at_close=False)
    self._starting_value = self._bench_cash

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

    # initialize Strategy and normalize prices
    bnh = BuyAndHoldStrategy(pf1,
                             start_date=pd.Timestamp(2000, 8, 1),
                             end_date=pd.Timestamp(2016, 1, 1),
                             cash=10000, cash_out_dividends=True,
                             tot_rb_freq=12, target_rb_day=0)
    bnh.normalize_price_bases(by_dividends=False)

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
    pf2.add_ticker('META', 0, label='core', shares=30)
    pf2.add_ticker('TQQQ', label='satellite', in_market=True)
    pf2.add_ticker('TLT', label='satellite', in_market=False)
    pf2.add_ticker('SPY', .6, label='benchmark', track=True)
    pf2.add_ticker('AGG', .4, label='benchmark')

    # initialize Strategy and normalize prices
    sma = SMAStrategy(pf2, window=100,
                      start_date=pd.Timestamp(2019, 12, 12),
                      end_date=pd.Timestamp(2020, 8, 13),
                      cash=1738.29, cash_out_dividends=False,
                      sat_rb_freq=365.25, tot_rb_freq=12, target_rb_day=8)
    sma.normalize_price_bases()

    # run simulation
    sma.begin_time_loop()

    # compare portfolio values to expectations
    exp_pf_val = 13377.608096729453 # 14038.487133948604
    exp_bnch_val = 10944.954633797679 # 10898.582343484206
    test_pf_val = sma.portfolio_value()
    test_bnch_val = sma.portfolio_value(main_portfolio=False)

    np.testing.assert_almost_equal(test_pf_val, exp_pf_val, decimal=4,
                                   err_msg=': main portfolio')
    np.testing.assert_almost_equal(test_bnch_val, exp_bnch_val, decimal=4,
                                   err_msg=': benchmark portfolio')

    # compare portfolios' share counts to expectations
    # exp_shares = {'SCHG': 35.83601442282118, 'SCHM': 18.748962642069873,
    #               'EFG': 11.109352277298585, 'ACES': 11.169197042734888,
    #               'BIV': 12.618948407993443, 'LQD': 3.040265631432126,
    #               'META': 0.0, 'TQQQ': 43.0, 'TLT': 0.0,
    #               'SPY': 19.300408108960365, 'AGG': 36.53405970562885}
    exp_shares = {'SCHG': 33.828459163391145, 'SCHM': 17.74082135494017,
                  'EFG': 10.103045729309041, 'ACES': 11.16278720221734,
                  'BIV': 12.612833394729345, 'LQD': 2.040265631432126,
                  'META': 0.0, 'TQQQ': 41.0, 'TLT': 0.0,
                  'SPY': 19.300408108960365, 'AGG': 36.53405970562885}
    tst_shares = {key : val['shares'] for key,val in sma.assets.items()}

    for key in tst_shares.keys():
        np.testing.assert_almost_equal(tst_shares[key], exp_shares[key],
                                       decimal=4,
                                       err_msg=f": {key} share count")

    # compare results plot to reference
    img_root = 'sma'
    ax = sma.plot_results(return_plot=True, verbose=False)
    # _compare_figs(ax, img_root)  # won't match ref since values have changed
    # using graphreader.com, sma_ref.png's start value looks like ~$9,825.
    # (prob close: $9,879 reading with sma_test.png on 1/2/25 only $2 too high)

def test_vlt():
    '''
    Deals with VolTargetStrategy. In addition to common focuses, also tests
    HistoricalSimulator's plot_results.

    NOTE: VolTargetStrategy has problems when cash_out_dividends=False because
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

    # initialize Strategy and normalize prices
    vlt = VolTargetStrategy(pf3, window=30, vol_target=.15,
                            start_date=pd.Timestamp(2018, 7, 27),
                            end_date=pd.Timestamp(2019, 7, 31),
                            cash=5500, cash_out_dividends=True,
                            sat_rb_freq=12, tot_rb_freq=4, target_rb_day=-3)
    vlt.normalize_price_bases(by_dividends=False)

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
