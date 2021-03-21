#!/usr/bin/python3
from simulator import HistoricalSimulator
import numpy as np
import pandas as pd

'''
This file holds example Strategy classes that inherit from HistoricalSimulator
and provide the logic for the decisions made in a portfolio's satellite
portion. The `on_new_day()` and `rebalance_satellite()` methods and `window`
attribute are required in order for HistoricalSimulator to work properly. As
long as you include those, it's also possible to write new Strategy classes
using your own trading logic.
'''

# save docstrings from required attributes/methods for convenience, if needed
WINDOW_DOCSTR = HistoricalSimulator.window.__doc__
ON_NEW_DAY_DOCSTR = HistoricalSimulator.on_new_day.__doc__
SAT_RB_DOCSTR = HistoricalSimulator.rebalance_satellite.__doc__
REFRESH_PARENT_DOCSTR = HistoricalSimulator.refresh_parent.__doc__

class TopXSplitStrategy(HistoricalSimulator):
    '''
    A Strategy class that buys into the "top X" performing assets over a
    previous number of days from a user-generated list. This is a "follow the
    demand" strategy that hopes any uptrends it finds continue to last.

    Arguments
    ---------

    Portfolio : `portfolio_maker.PortfolioMaker`, required
        A PortfolioMaker instance whose `assets` attribute contains your desired
        assets, fractions, and categories. **Its `sat_frac` attribute must be 0
        for it to work with this Strategy.**

    window : float, optional
        The number of previous days to consider for the asset performance
        calculation. For example, `window=20` considers the percent change
        in an asset's closing price from 20 market days ago and the last market
        day before today.

    top_x : float, optional
        The number of top-perfoming assets to include in the portfolio on
        rebalances. For example, `top_x=3` will split the portfolio between the
        top 3 performing assets in the last `window` days.

    **kwargs : See "Arguments" in the docstring for HistoricalSimulator, but
    make sure to not use `sat_rb_freq` as it's incompatible with this Strategy.
    '''

    def __init__(self, Portfolio, window=30, top_x=3, **kwargs):
        # save number of days in window
        self.window = window

        # pre-validate kwargs, then set up HistoricalSimulator instance
        kwargs = self._pre_validate_args(Portfolio, **kwargs)
        super().__init__(Portfolio, **kwargs)

        # then, set strategy-specific attributes
        self.top_x = top_x
        self.hist_pct_changes = self._calc_pct_changes()

        # use existing docstrings for unchanged abstract methods
        #self.on_new_day.__func__.__doc__ = ON_NEW_DAY_DOCSTR
        self.rebalance_satellite.__func__.__doc__ = SAT_RB_DOCSTR
        self.refresh_parent.__func__.__doc__ = REFRESH_PARENT_DOCSTR

    def _pre_validate_args(self, Portfolio, **kwargs):
        '''
        Ensure that the keyword arguments meant for HistoricalSimulator are
        valid for this type of Strategy. (For TopXSplitStrategy, that means
        disallowing satellite assets.)

        See the  __init__() docstring for more on `Portfolio` and **kwargs.
        '''
        # satellite fraction must be 0
        if Portfolio.sat_frac != 0:
            raise ValueError("This Strategy can't have satellite assets, so "
                             "the Portfolio object's `sat_frac` must equal 0.")
            # this being the case, even if there are satellite assets in
            # Portfolio, they won't affect the simulation

        # if tot_rb_freq is present, set sat_rb_freq equal to it
        # to avoid satellite-only rebalances
        if 'tot_rb_freq' in kwargs:
            kwargs['sat_rb_freq'] = kwargs['tot_rb_freq']
            # warn the user that this is happening?
        elif 'sat_rb_freq' in kwargs:
            raise ValueError("This Strategy doesn't use satellite assets, "
                             "so `sat_rb_freq` should not be set.")
        return kwargs

    def _calc_pct_changes(self):
        '''
        Called from __init__() of TopXSplitStrategy.

        Calculate the rolling historical `window`-day percent changes between
        closing prices for each 'core'-labeled ticker in self.assets.

        Returns a dictionary with the same keys as self.assets. Each key
        contains a Series of rolling percent changes whose indices match up with
        self.active_dates.

        For a 20-day period, the value at index 21 of any key is the percent
        change in closing price between days 1 and 20.
        '''

        pct_changes = {}

        for nm in self.core_names:
            # (past method of building dataFrame; included current day's close)
            #prices = self.assets[nm]['df']['adjClose'].pct_change(self.window)
            #pct_changes[nm] = prices[self.start_date:]

            # get array of all `window`-day percent changes for this asset
            all_closes = self.assets[nm]['df']['adjClose']
            all_changes = all_closes.pct_change(self.window).shift()

            # shift them forward by one day so the calculation is totally
            # backward-looking instead of inclusive of the current day
            # (should be NaN-safe due to buffer_days in HistoricalSimulator)
            pct_changes[nm] = all_changes[self.start_date:]

        return pct_changes

    def refresh_parent(self): # docstring set in __init__()
        self.hist_pct_changes = self._calc_pct_changes()

    def on_new_day(self):
        '''
        Called in HistoricalSimulator.begin_time_loop().

        Tracks the best-performing `top_x` assets by percent change in the last
        `window` days. (Note that the portfolio's composition will only change
        on rebalance days.)

        Returns
        -------

        Nothing.
        '''
        frac = 1 / self.top_x

        # get today's percent change for each ticker; rank them best to worst
        changes = {tk: self.hist_pct_changes[tk].loc[self.today]
                   for tk in self.core_names}
        ranked = sorted(changes.items(), key=lambda itms: itms[1], reverse=True)

        # separate top performing tickers from the rest
        top_tks = dict(ranked[:self.top_x]).keys()
        oth_tks = dict(ranked[self.top_x:]).keys()

        # set intended portfolio fraction for top performing tickers
        for tk in top_tks:
            self.assets[tk]['fraction'] = frac

        # exclude other tickers from the portfolio
        for tk in oth_tks:
            self.assets[tk]['fraction'] = 0

    def rebalance_satellite(self, day, verbose=False):
        # docstring set in __init__()
        return []

class BuyAndHoldStrategy(HistoricalSimulator):
    '''
    A Strategy class for buying an all-core portfolio of assets that only seeks
    to invest in them at their original target fractions over time.

    Arguments
    ---------

    Portfolio : `portfolio_maker.PortfolioMaker`, required
        A PortfolioMaker instance whose `assets` attribute contains your desired
        assets, fractions, and categories. **Its `sat_frac` attribute must be 0
        for it to work with this Strategy.**

    **kwargs : See "Arguments" in the docstring for HistoricalSimulator, but
    make sure to not use `sat_rb_freq` as it's incompatible with this Strategy.
    '''
    def __init__(self, Portfolio, **kwargs):
        kwargs = self._pre_validate_args(Portfolio, **kwargs)

        self.window = 0 # no past price info is needed for this Strategy

        super().__init__(Portfolio, **kwargs)

        # use existing docstrings for unchanged abstract methods
        self.on_new_day.__func__.__doc__ = ON_NEW_DAY_DOCSTR
        self.rebalance_satellite.__func__.__doc__ = SAT_RB_DOCSTR
        self.refresh_parent.__func__.__doc__ = REFRESH_PARENT_DOCSTR

    def _pre_validate_args(self, Portfolio, **kwargs):
        '''
        Ensure that the keyword arguments meant for HistoricalSimulator are
        valid for this type of Strategy. (For BuyAndHoldStrategy, that means
        disallowing satellite assets.)

        See the  __init__() docstring for more on `Portfolio` and **kwargs.
        '''
        # satellite fraction must be 0
        if Portfolio.sat_frac != 0:
            raise ValueError("This Strategy can't have satellite assets, so "
                             "the Portfolio object's `sat_frac` must equal 0.")
            # this being the case, even if there are satellite assets in
            # Portfolio, they won't affect the simulation

        # if tot_rb_freq is present, set sat_rb_freq equal to it
        # to avoid satellite-only rebalances
        if 'tot_rb_freq' in kwargs:
            kwargs['sat_rb_freq'] = kwargs['tot_rb_freq']
            # warn the user that this is happening?
        elif 'sat_rb_freq' in kwargs:
            raise ValueError("This Strategy doesn't use satellite assets, "
                             "so `sat_rb_freq` should not be set.")
        return kwargs

    def refresh_parent(self):
        # docstring set in __init__()
        pass

    def on_new_day(self):
        # docstring set in __init__()
        pass

    def rebalance_satellite(self, day, verbose=False):
        # docstring set in __init__()
        return []

class SMAStrategy(HistoricalSimulator):
    # **target_sma_streak, sma_threshold, & retreat_period should be args**
    '''
    A Strategy that assigns the entire satellite portion of a portfolio to
    either the in-market or out-of-market asset depending on whether a tracked
    asset's current price is above a multiple of its X-day (see `window` in
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

    window : float, optional
        An attribute representing the number of days of data needed before a
        Strategy class can begin trading. For example, a 200-day simple moving
        average needs `window=200`.

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
    def __init__(self, Portfolio, window=200, **kwargs):
        # save number of days in window
        self.window = window

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
        self.hist_smas = self._calc_rolling_smas()

        # daily indicators related to tracked asset
        self.sma_streak = 0 # how many consecutive days of price above SMA?
        self.sma_threshold = 1.01 # minimum safe multiple of SPY for entering

        # other daily indicators for making in or out decision for satellite
        self.can_enter = True # can we enter the market?
        self.retreat_period = 60 # minimum days to remain out after retreating
        self.days_out = 0 # how many consecutive days have you been in retreat?

        # use existing docstrings for unchanged abstract methods
        self.on_new_day.__func__.__doc__ = ON_NEW_DAY_DOCSTR
        #self.rebalance_satellite.__func__.__doc__ = SAT_RB_DOCSTR
        self.refresh_parent.__func__.__doc__ = REFRESH_PARENT_DOCSTR

    def _calc_rolling_smas(self):
        '''
        Called from __init__() of SMAStrategy.

        Calculates the historical rolling `window`-day simple moving average of
        closing price for each ticker in self.assets.

        Returns a dictionary with the same keys as self.assets. Each key
        contains a Series of rolling simple moving averages whose indices
        match up with self.active_dates.

        For a 20-day SMA, the value at index 21 of any key is the average close
        from days 1-20.

        (Sidenote: This method worked differently in commits up to March 2021,
        but this configuration takes about the same runtime as the old
        _calc_rolling_smas() and calc_moving_avg() combination even while adding
        a shift. The resulting numbers are different at the 1e-12 level, but
        that's probably too small to worry over for this application.)
        '''
        smas = {}
        for nm in self.assets.keys():
            # make `window`-day collections of closing prices for all dates
            rolled = self.assets[nm]['df']['adjClose'].rolling(self.window,
                                                               self.window)

            # calculate means; then shift them forward by one day so they're
            # totally backward-looking instead of inclusive of the current day
            # (should be NaN-safe due to buffer_days in HistoricalSimulator)
            smas[nm] = rolled.mean().shift().loc[self.start_date:]

        return smas

    def refresh_parent(self): # docstring set in __init__()
        self.hist_smas = self._calc_rolling_smas()

    def on_new_day(self): # docstring set in __init__()
        # get YESTERDAY's close and the tracked asset's current `window`-day SMA
        prev_close = self.assets[self.track_tick]['df'].loc[self.prev_day,
                                                            'adjClose']
        prev_sma = self.hist_smas[self.track_tick].loc[self.today]

        # check if close was over SMA threshold and adjust streak counters
        if prev_close >= prev_sma * self.sma_threshold: # streak builds
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

    window : float, optional
        An attribute representing the number of days of data needed before a
        Strategy class can begin trading. For example, using 30-day volatility
        requires that `window=30`. [default: 30]

    vol_target : float, optional
        The target `window`-day annualized volatility for the satellite
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
    def __init__(self, Portfolio, window=30, vol_target=.15, **kwargs):
        # save number of days in window for calculating historical volatility
        self.window = window

        super().__init__(Portfolio, **kwargs)

        # then, set strategy-specific attributes
        # set target `window`-day annualized volatility for satellite portfolio
        self.vol_target = vol_target

        # calculate assets' rolling standard deviations from daily closes
        self.hist_stds = self._calc_rolling_stds()

        # calculate inter-asset correlations
        self.hist_corrs = self._calc_correlations()

        # use existing docstrings for unchanged abstract methods
        self.on_new_day.__func__.__doc__ = ON_NEW_DAY_DOCSTR
        #self.rebalance_satellite.__func__.__doc__ = SAT_RB_DOCSTR
        self.refresh_parent.__func__.__doc__ = REFRESH_PARENT_DOCSTR

    def _calc_correlations(self):
        '''
        Called from __init__() of VolTargetStrategy.

        Calculates historical rolling `window`-day correlations between all
        tickers in self.assets.

        Returns a nested dictionary that takes two asset labels as indices
        and gives back a Series containing correlation values between them.

        For example, corrs['TICK1']['TICK2'] gives a Series of correlations
        between TICK1 and TICK2. For a 20-day window, the value at index 21 is
        their correlation from days 1-20. The Series' index contains the same
        dates as self.active_dates and the results of self._get_moving_stats().
        '''
        all_keys = list(self.assets.keys())
        rem_keys = all_keys.copy()

        # create nested dictionary with all asset pairs (Series of ones by
        # default to cover the same-asset scenario up front)
        ones = pd.Series(1., index=self.active_dates)
        corrs = {tk1: {tk2: ones for tk2 in all_keys} for tk1 in all_keys}

        # fill the dictionary with each Series of inter-asset correlations
        for nm1 in all_keys:
            rem_keys.remove(nm1)
            for nm2 in rem_keys:
                p1 = self.assets[nm1]['df']['adjClose']
                p2 = self.assets[nm2]['df']['adjClose']

                # calculate correlations and shift them forward by one day to
                # keep them backward-looking and exclusive of the current day
                # (should be NaN-safe due to buffer_days in HistoricalSimulator)
                corr = p1.rolling(self.window).corr(p2).shift()
                # (correlation from Pearson product-moment corr. matrix)
                # np.corrcoef(in, out) & np.cov(in,out) / (stddev_1 * stddev_2)
                # give the same result (index [0][1]) when the stddevs' ddof = 1

                corrs[nm1][nm2] = corrs[nm2][nm1] = corr.loc[self.start_date:]

        return corrs

    def _calc_rolling_stds(self):
        '''
        Called from __init__() of VolTargetStrategy.

        Calculates historical rolling `window`-day standard deviations
        (annualized) for all tickers in self.assets.

        Returns a dictionary with the same keys as self.assets. Each key
        contains a Series of rolling standard deviations whose indices match up
        with self.active_dates.

        For a 20-day window, the value at index 21 of any key is that asset's
        annualized standard deviation from days 1-20.
        '''
        stds = {}
        for nm in self.assets.keys():
            prices = self.assets[nm]['df']['adjClose']

            # record asset's daily logartihmic returns
            log_ret = np.log(prices).diff().fillna(0) # (change 0th entry to 0)

            # collect active dates' rolling `window`-day standard deviations,
            # shifting them forward one day to keep them backward-looking
            # (should be NaN-safe due to buffer_days in HistoricalSimulator)
            devs = log_ret.rolling(self.window).std().shift()[self.start_date:]

            # save a Series containing annualized standard deviations
            stds[nm] = devs * np.sqrt(252)
            # tried a numpy-only solution but the speed gain was minimal:
            # https://stackoverflow.com/questions/43284304/

        return stds

    def refresh_parent(self):
        # docstring set in __init__()
        self.hist_corrs = self._calc_correlations()
        self.hist_stds = self._calc_rolling_stds()

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

        # Get `window`-day correlation and annualized standard deviation values
        correlation = self.hist_corrs[in_mkt_tick][out_mkt_tick][self.today]
        stddev_in = self.hist_stds[in_mkt_tick][self.today]
        stddev_out = self.hist_stds[out_mkt_tick][self.today]

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
            my_pr('in_mkt partial shares (if any) will be sold!')
        in_mkt_delta = in_mkt_ideal - in_mkt_sh

        out_mkt_ideal = int((can_spend - in_mkt_pr * in_mkt_ideal) / out_mkt_pr)
        if out_mkt_ideal == 0 and self.reinvest_dividends == True:
            # also sell partial shares
            out_mkt_sh = self.assets[out_mkt_tick]['shares']
            my_pr('out_mkt partial shares (if any) will be sold!')
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
