#!/usr/bin/python3
from simulator import HistoricalSimulator
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

'''
This file holds example Strategy classes that inherit from HistoricalSimulator
and provide the logic for the decisions made in a portfolio's satellite
portion. The `on_new_day()` and `rebalance_satellite()` methods and `burn_in`
attribute are required in order for HistoricalSimulator to work properly. As
long as you include those, it's also possible to write new Strategy classes
using your own trading logic.
'''

# save docstrings from required attributes/methods for convenience, if needed
BURN_IN_DOCSTR = HistoricalSimulator.burn_in.__doc__
ON_NEW_DAY_DOCSTR = HistoricalSimulator.on_new_day.__doc__
SAT_RB_DOCSTR = HistoricalSimulator.rebalance_satellite.__doc__
REFRESH_PARENT_DOCSTR = HistoricalSimulator.refresh_parent.__doc__

class SMAStrategy(HistoricalSimulator):
    # **target_sma_streak, sma_threshold, & retreat_period should be args**
    '''
    A Strategy that assigns the entire satellite portion of a portfolio to
    either the in-market or out-of-market asset depending on whether a tracked
    asset's current price is above a multiple of its X-day (see `burn_in` in
    "Arguments") simple moving average, or SMA.

    Method self.on_new_day() makes this comparison every day and changes
    self.can_enter so the appropriate move is made during the next rebalance.

    At the moment, requirements for going in are that the tracked asset's price
    has been above a multiple of its SMA for at least 3 days (self.sma_streak),
    and, if the satellite portion is currently out, that it has been in retreat
    for at least 60 days (self.retreat_period). (The latter requirement is
    needed because volatility is least predictable when an index is around its
    SMA, and volatility spikes lead to quick losses with leveraged instruments.)
    See "Motivation" for sources.

    Arguments
    ---------

    Portfolio : `portfolio_maker.PortfolioMaker`, required
        A PortfolioMaker instance whose `assets` attribute contains your desired
        assets, fractions, and categories. SPECIAL TO THIS STRATEGY: Make sure
        one of the assets has a 'track' key that equals True, as that will be
        the asset whose simple moving average is used to make decisions in
        self.on_new_day().

    burn_in : float, optional
        An attribute representing the number of days of data needed before a
        Strategy class can begin trading. For example, a 200-day simple moving
        average needs `burn_in=200`.

    **kwargs : See "Arguments" in the docstring for HistoricalSimulator.

    Motivation
    ----------

    Leveraged instruments are alluring because they mutiply returns when market
    conditions are positive. However, they also multiply risk by compounding the
    effects of negative outcomes like market decline and volatility. Any market
    decline is multiplied in value, and volatility erodes gains and makes
    leveraged instruments underperform their indices over time.

    No one has figured out how to predict returns, but month to month volatility
    actually has a slight correlation at .35.
    (See: https://www.lazardassetmanagement.com/docs/sp0/22430/predictingvolatility_lazardresearch.pdf)

    With this knowledge, we can use the past to predict high volatility periods,
    eliminating some of the risk of using leverage without removing its upside.

    Tracking a representative index's (e.g., SPY) daily price versus its X-day
    SMA is a proxy for measuring market volatility. When the daily price has
    been above the SMA for a prolonged period, it portends low volatility.
    (See: https://poseidon01.ssrn.com/delivery.php?ID=083008085091002109125001065101006023014014073092023027101003102086027067106094091125028029021119038017111107031120127020001017122075014092018092006113115081066001022040064023094031090122120029088116091119011001124126118021016024000020069099092010116064)

    Moving in and out of a leveraged instrument based on this signal won't
    always prevent losses, but repeated trials should show that it works well
    enough to beat conventional SPY/AGG portfolios by considerable amounts in
    almost any 5+ year period, bear or bull.
    (Also: inspired by https://seekingalpha.com/article/4230171)
    '''
    def __init__(self, Portfolio, burn_in=200, **kwargs):
        # save number of burn in days
        self.burn_in = burn_in

        super().__init__(Portfolio, **kwargs)

        # save ticker whose SMA will be tracked
        to_track = [key for key, val in self.assets.items() if 'track' in val]
        if len(to_track) == 1:
            self.track_tick = to_track.pop()
        else:
            raise KeyError('For this strategy, one of the tickers in the '
                           "assets dict must have a 'track' key set to True. "
                           f"You have {len(to_track)} tickers with this key.")

        # calculate assets' rolling simple moving average from daily closes
        self.smas = self._calc_rolling_smas()

        # daily indicators related to tracked asset
        self.sma_streak = 0 # how many consecutive days of price above SMA?
        self.sma_threshold = 1.01 # minimum safe multiple of SPY for entering

        # other daily indicators for making in or out decision for satellite
        self.can_enter = True # can we enter the market?
        self.retreat_period = 60 # minimum days to remain out after retreating
        self.days_out = 0 # how many consecutive days have you been in retreat?

        # use existing docstrings for on_new_day() and rebalance_satellite()
        self.on_new_day.__func__.__doc__ = ON_NEW_DAY_DOCSTR
        #self.rebalance_satellite.__func__.__doc__ = SAT_RB_DOCSTR
        self.refresh_parent.__func__.__doc__ = REFRESH_PARENT_DOCSTR

    def calc_mv_avg(self, prices):
        '''
        Called from self._calc_rolling_smas().

        A faster method of calculating moving averages than slicing + np.mean.
        This method is also about an order of magnitude faster than
        prices.rolling(window=self.burn_in).mean()[self.burn_in-1:].values.

        Numerical precision apparently breaks down when len(prices) > 1e7, but
        that's almost 30,000 years worth of days, so no worries currently. May
        have to shift if we use price data with finer resoluion in the future.

        Arguments
        ---------

        prices : pandas.core.series.Series, required
            The selection of prices to be averaged.

        Found here, along with concerns: https://stackoverflow.com/a/27681394
        Simplified here: https://stackoverflow.com/a/43286184
        '''
        #cumsum = np.cumsum(np.insert(prices.values, 0, 0))
        cumsum = np.cumsum(prices)
        #return (cumsum[burn_in:] - cumsum[:-burn_in]) / float(burn_in)
        cumsum[self.burn_in:] = (cumsum[self.burn_in:].values
                                 - cumsum[:-self.burn_in].values)

        active_ind = cumsum.index.get_loc(self.start_date)
        return cumsum[active_ind - 1:] / self.burn_in

    def _calc_rolling_smas(self):
        '''
        Called from __init__() of SMAStrategy.

        Calculate rolling simple moving average of closing price for each
        ticker. The length of a period is `burn_in`.

        Returns a dictionary with the same keys as self.assets. Each key
        contains an array of rolling simple moving averages whose indices match
        up with self.active_dates.
        '''
        smas = {}
        for nm in self.assets.keys():
            prices = self.assets[nm]['df']['adjClose']

            # get array of all `burn_in`-day simple moving averages
            smas[nm] = self.calc_mv_avg(prices)

        return smas

    def refresh_parent(self): # docstring set in __init__()
        self.smas = self._calc_rolling_smas()

    def on_new_day(self): # docstring set in __init__()
        tracked_price = self.assets[self.track_tick]['df'].loc[self.today,
                                                               'adjClose']
        tracked_sma = self.smas[self.track_tick].loc[self.today]

        # check if SMA is over our threshold and adjust streak counters
        if tracked_price >= tracked_sma * self.sma_threshold: # streak builds
            self.sma_streak += 1
            if self.sma_streak >= 3 and self.days_out >= self.retreat_period:
                # if we were out, get back in
                self.can_enter = True
            elif self.days_out < self.retreat_period: # if we're already in...
                self.days_out += 1
        else: # streak broken, get/stay out of the market
            if self.can_enter == True: # if we were in...
                self.days_out = 0
            else: # if we were already out...
                self.days_out += 1
            self.can_enter = False
            self.sma_streak = 0

    def rebalance_satellite(self, day, verbose=False):
        '''
        Called in HistoricalSimulator.rebalance_portfolio() or
        HistoricalSimulator.begin_time_loop().

        During a rebalance, moves the entire satellite portion of the portfolio
        into either the in-market or out-of-market asset, depending on whether
        `self.can_enter` is True or False.

        On satellite-only rebalances, the method completes any needed
        transactions itself. During whole portfolio rebalances, it returns a
        list of share changes for the in-market or out-of-market assets.

        Arguments
        ---------

        day : `pandas.Timestamp` or `datetime.datetime`, required
            The simulation's current date. Used to find whether this rebalance
            is satellite-only or for the total portfolio.

        verbose : boolean, optional
            Controls whether or not to print any debugging information you
            choose to include in this method. [default: False]
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)
        my_pr(f"satellite rb; sat_only is {self.rb_info.loc[day, 'sat_only']}; "
              f"${self.cash:.2f} in account")

        # exit if there are no satellite assets
        if len(self.sat_names) == 0:
            # empty list needed when called from self.rebalance_portfolio()
            return []

        # (purchases will be made using TODAY's PRICES, NOT yesterday's closes)
        in_mkt_tick = self.sat_names[0]
        in_mkt_pr = self.assets[in_mkt_tick]['df'].loc[self.today, 'adjOpen']
        in_mkt_sh = self.assets[in_mkt_tick]['shares']

        out_mkt_tick = self.sat_names[-1]
        out_mkt_pr = self.assets[out_mkt_tick]['df'].loc[self.today, 'adjOpen']
        out_mkt_sh = self.assets[out_mkt_tick]['shares']

        total_mkt_val = self.portfolio_value(at_close=False)

        # with enough consecutive days above SMA (and a buffer, if necessary)...
        if self.can_enter:
            # switch to in-market asset if not already there
            if out_mkt_sh > 0:
                # liquidate out_mkt holdings
                out_mkt_delta = -out_mkt_sh
                out_mkt_cash = out_mkt_sh * out_mkt_pr

                if self.rb_info.loc[day, 'sat_only']: # buy in w/ all avail. $$
                    my_pr('1: liq out, buy in')
                    in_mkt_delta = int((self.cash + out_mkt_cash) // in_mkt_pr)
                else: # if total rebalance, re-weight and return shares to buy
                    my_pr('2: liq out, balance in')
                    in_mkt_delta = (total_mkt_val * self.sat_frac) // in_mkt_pr
                    return [in_mkt_delta, out_mkt_delta]
            # if already invested in in-market asset...
            else:
                if self.rb_info.loc[day, 'sat_only']: # already in; no change
                    my_pr('3: already in, remain in')
                    in_mkt_delta = out_mkt_delta = 0
                else: # if total rebalance, find and return net share change
                    my_pr('4: already in, balance in')
                    out_mkt_delta = 0
                    in_hold_val = in_mkt_sh * in_mkt_pr
                    in_mkt_delta = (total_mkt_val * self.sat_frac
                                    - in_hold_val) // in_mkt_pr
                    return [in_mkt_delta, out_mkt_delta]
        # without enough consecutive days above SMA or enough buffer...
        elif not self.can_enter:# and self.active:
            # retreat to out of market asset if not already there
            if in_mkt_sh > 0:
                # liquidate in_mkt holdings
                in_mkt_delta = -in_mkt_sh
                in_mkt_cash = in_mkt_sh * in_mkt_pr

                if self.rb_info.loc[day, 'sat_only']: # retreat w/ all avail. $$
                    my_pr('5: liq in, buy out')
                    out_mkt_delta = int((self.cash + in_mkt_cash) // out_mkt_pr)
                else: # if total rebalance, find and return net share change
                    my_pr('6: liq in, balance out')
                    out_mkt_delta = (total_mkt_val*self.sat_frac) // out_mkt_pr
                    return [in_mkt_delta, out_mkt_delta]
            # if already invested in out-market asset...
            else:
                if self.rb_info.loc[day, 'sat_only']: # already out; no change
                    my_pr('7: already out, remain out')
                    in_mkt_delta = out_mkt_delta = 0
                else: # if total rebalance, find and return net share change
                    my_pr('8: already out, balance out')
                    in_mkt_delta = 0
                    out_hold_val = out_mkt_sh * out_mkt_pr
                    out_mkt_delta = (total_mkt_val * self.sat_frac
                                     - out_hold_val) // out_mkt_pr
                    return [in_mkt_delta, out_mkt_delta]

        # if this is a satellite-only rebalance, complete the trades
        names = np.array(self.sat_names)
        deltas = np.array([in_mkt_delta, out_mkt_delta])
        self.make_rb_trades(names, deltas, verbose=verbose)

class VolTargetStrategy(HistoricalSimulator):
    '''
    A Strategy that tries to balance the weights of a satellite portfiolio's
    in-market and out-of-market assets so that the portion as a whole has a
    specific, targeted volatility (a.k.a. standard deviation). The in-market
    asset should be the more volatile of the two, while the out-of-market
    asset should provide stability.

    The weights are checked and, if necessary, changed any time the satellite
    portfolio is rebalanced (individually or as part of a total portfolio
    rebalance). See the thinking behind this Strategy in "Motivation."

    Arguments
    ---------

    Portfolio : `portfolio_maker.PortfolioMaker`, required
        A PortfolioMaker instance whose `assets` attribute contains your desired
        assets, fractions, and categories.

    burn_in : float, optional
        An attribute representing the number of days of data needed before a
        Strategy class can begin trading. For example, using 30-day volatility
        requires that `burn_in=30`. [default: 30]

    vol_target : float, optional
        The target `burn_in`-day annualized volatility for the satellite
        portfolio. Takes decimal values, not percentages. [default: .15]

    **kwargs : See "Arguments" in the docstring for HistoricalSimulator.

    Motivation
    ----------

    Leveraged instruments are alluring because they mutiply returns when market
    conditions are positive. However, they also multiply risk by compounding the
    effects of negative outcomes like market decline and volatility. Any market
    decline is multiplied in value, and volatility erodes gains and makes
    leveraged instruments underperform their indices over time.

    No one has figured out how to predict returns, but month to month volatility
    actually has a slight correlation at .35.
    (See: https://www.lazardassetmanagement.com/docs/sp0/22430/predictingvolatility_lazardresearch.pdf)

    If you've already defined a level of volatility you're comfortable with
    (`vol_target`), you can determine the combination of the two assets that
    most closely approached it over the previous month, use those allocations
    over the next month, and be reasonably confident that the combined
    volatility won't stray too far from your target.

    Of course, this is not always the case, but this strategy's gambit is that
    targeting volatility reduces risk compared to a standard SPY/AGG portfolio
    while still outperforming it most of the time. Let's put it to the test.
    '''
    def __init__(self, Portfolio, burn_in=30, vol_target=.15, **kwargs):
        # save number of burn in days
        self.burn_in = burn_in # volatility period

        super().__init__(Portfolio, **kwargs)

        # then, set strategy-specific attributes
        # set target `burn_in`-day annualized volatility for satellite portfolio
        self.vol_target = vol_target

        # calculate assets' rolling standard deviations from daily closes
        self.stds = self._calc_rolling_stds()

        # calculate inter-asset correlations
        self.corrs = self._calc_correlations()

        # use existing docstrings for on_new_day() and rebalance_satellite()
        self.on_new_day.__func__.__doc__ = ON_NEW_DAY_DOCSTR
        #self.rebalance_satellite.__func__.__doc__ = SAT_RB_DOCSTR
        self.refresh_parent.__func__.__doc__ = REFRESH_PARENT_DOCSTR

    def _calc_rolling_stds(self):
        '''
        Called from __init__() of VolTargetStrategy.

        Calculate rolling standard deviation for each ticker. The length of a
        rolling period is self.burn_in.

        Returns a dictionary with the same keys as self.assets. Each key
        contains an array of rolling standard deviations whose indices match up
        with self.active_dates.
        '''
        stds = {}
        for nm in self.assets.keys():
            prices = self.assets[nm]['df']['adjClose']

            # record asset's daily logartihmic returns
            log_ret = np.log(prices).diff().fillna(0) # (change 0th entry to 0)

            # collect rolling `burn_in`-day standard deviations; slice out nans
            devs = log_ret.rolling(self.burn_in).std()[self.burn_in - 1:]

            # save the array of standard deviations after annualizing them
            stds[nm] = devs.values * np.sqrt(252)
            # tried a numpy-only solution but the speed gain was minimal:
            # https://stackoverflow.com/questions/43284304/

        return stds

    def _calc_correlations(self):
        '''
        Called from __init__() of VolTargetStrategy.

        Calculate rolling inter-asset correlations for all tickers in
        self.assets. Returns a DataFrame that takes two asset labels as indices
        and gives back an array of correlation values between those assets.

        For example, corrs['TICK1']['TICK2'] (provide the ticker strings) gives
        an array of correlations. The indices of the array match up with the
        results of self._get_moving_stats() and self.active_dates.

        At the moment, an asset's correlation with itself is NaN instead of an array of 1s. It should save a little time in the loop, but does it matter?
        '''
        all_keys = list(self.assets.keys())
        rem_keys = all_keys.copy()
        corrs = pd.DataFrame(columns=all_keys, index=all_keys)

        for nm1 in all_keys:
            rem_keys.remove(nm1)
            for nm2 in rem_keys:
                p1 = self.assets[nm1]['df']['adjClose']
                p2 = self.assets[nm2]['df']['adjClose']

                corr = p1.rolling(self.burn_in).corr(p2)[self.burn_in-1:].values
                # (correlation from Pearson product-moment corr. matrix)
                # np.corrcoef(in, out) & np.cov(in,out) / (stddev_1 * stddev_2)
                # give the same result (index [0][1]) when the stddevs' ddof = 1

                corrs[nm1][nm2] = corr
                corrs[nm2][nm1] = corr

        return corrs

    def refresh_parent(self):
        # docstring set in __init__()
        pass

    def on_new_day(self):
        # docstring set in __init__()
        pass

    def rebalance_satellite(self, day, verbose=False):
        '''
        Adjusts the in- and out-of-market portions of the satellite to target
        the user's chosen volatility (self.vol_target).

        On satellite-only rebalances, the method completes any needed
        transactions itself. During whole portfolio rebalances, it returns a
        list of share changes for the in-market or out-of-market assets.

        Arguments
        ---------

        day : `pandas.Timestamp` or `datetime.datetime`, required
            The simulation's current date. Used to find whether this rebalance
            is satellite-only or for the total portfolio.

        verbose : boolean, optional
            Controls whether or not to print any debugging information you
            choose to include in this method. [default: False]
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)
        my_pr(f"satellite rb; sat_only is {self.rb_info.loc[day, 'sat_only']}; "
              f"${self.cash:.2f} in account")

        # exit if there are no satellite assets
        if len(self.sat_names) == 0:
            # empty list needed when called from self.rebalance_portfolio()
            return []

        # What are my current satellite holdings worth?
        # (purchases will be made using TODAY's PRICES, NOT yesterday's closes)
        in_mkt_tick = self.sat_names[0]
        in_mkt_pr = self.assets[in_mkt_tick]['df'].loc[self.today, 'adjOpen']
        in_mkt_sh = int(self.assets[in_mkt_tick]['shares'])
        in_mkt_val = in_mkt_pr * in_mkt_sh

        out_mkt_tick = self.sat_names[-1]
        out_mkt_pr = self.assets[out_mkt_tick]['df'].loc[self.today, 'adjOpen']
        out_mkt_sh = int(self.assets[out_mkt_tick]['shares'])
        out_mkt_val = out_mkt_pr * out_mkt_sh

        # What can i spend during this rebalance?
        total_mkt_val = self.portfolio_value(at_close=False)
        can_spend = ((in_mkt_val + out_mkt_val + self.cash)
                     if self.rb_info.loc[day, 'sat_only']
                     else total_mkt_val * self.sat_frac)
        # (available sum to spend depends on rebalance type)

        # Get `burn_in`-day correlation and annualized standard deviation values
        correlation = self.corrs[in_mkt_tick][out_mkt_tick][ind_active]
        stddev_in = self.stds[in_mkt_tick][ind_active]
        stddev_out = self.stds[out_mkt_tick][ind_active]

        # Test different in/out weights and record resulting volatilites
        n_tests = 21
        vol_tests = np.zeros(n_tests)
        fracs = np.linspace(1, 0, n_tests, endpoint=True)
        for i, frac_in in enumerate(fracs):
            frac_out = 1 - frac_in

            # calc two asset annualized volatility for these weights
            # (gives a decimal, not a percentage!)
            curr_vol = np.sqrt((frac_in**2 * stddev_in**2)
                               + (frac_out**2 * stddev_out**2)
                               + (2 * frac_in * frac_out
                                  * stddev_in * stddev_out * correlation))

            # ensure this index isn't chosen if curr_vol == 0
            vol_tests[i] = curr_vol if curr_vol > 0 else 1e6

        # Isolate the weighting that comes closet to our target volatility
        # (ties will go to the combination with a higher in_mkt fraction)
        new_frac_in = fracs[np.argmin(np.abs(vol_tests - self.vol_target))]
        new_frac_out = 1 - new_frac_in

        my_pr(', '.join([f"{k[0]*1e2:.0f}%: {k[1]:.2f}"
                        for k in np.stack((fracs, vol_tests), axis=1)]))
        my_pr(f"vol target: {self.vol_target:.2f}. frac in: {new_frac_in:.2f}")
        my_pr(f"current shares: {self.assets[in_mkt_tick]['shares']} in, "
              f"{self.assets[out_mkt_tick]['shares']} out")

        # Find the resulting net change in shares held for both assets
        # (This strategy is vulnerable to the odd floating point "error" that
        #  uses all available cash plus $0.000000001 and throws an error. Using
        #  decimal.Decimal on self.cash/bench_cash, all entries in
        #  self.assets[tk]['shares'], and can_spend & in/out_mkt_pr here can
        #  prevent that, but it's slower than using floats...)
        # (Also, for non-mutual funds, when reinvest_dividends=True, need to
        #  disallow all partial share purchases. Need to do the same for sales,
        #  **except** when all shares are being sold.)
        max_in_sh = int(can_spend / in_mkt_pr)
        poss_in_sh = np.linspace(0, max_in_sh, max_in_sh + 1)
        poss_in_fracs = (in_mkt_pr * poss_in_sh) / can_spend

        in_mkt_ideal = np.argmin(np.abs(poss_in_fracs - new_frac_in))
        if in_mkt_ideal == 0 and self.reinvest_dividends == True:
            # also sell partial shares
            in_mkt_sh = self.assets[in_mkt_tick]['shares']
            print('in_mkt partial shares (if any) will be sold!')
        in_mkt_delta = in_mkt_ideal - in_mkt_sh

        out_mkt_ideal = int((can_spend - in_mkt_pr * in_mkt_ideal) / out_mkt_pr)
        if out_mkt_ideal == 0 and self.reinvest_dividends == True:
            # also sell partial shares
            out_mkt_sh = self.assets[out_mkt_tick]['shares']
            print('out_mkt partial shares (if any) will be sold!')
        out_mkt_delta = out_mkt_ideal - out_mkt_sh

        # Find the resulting net change in shares held
        my_pr(f"out_mkt_delta: {out_mkt_delta}, out_mkt_val: {out_mkt_val}, out_mkt_pr: {out_mkt_pr}")
        deltas = [in_mkt_delta, out_mkt_delta]

        # Return share change values if this is a total-portfolio rebalance
        if not self.rb_info.loc[day, 'sat_only']:
            return deltas
        else: # If this is a satellite-only rebalance, make the trades
            names = np.array(self.sat_names)
            deltas = np.array(deltas)
            self.make_rb_trades(names, deltas, verbose=verbose)
