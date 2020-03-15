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
portion. The `on_new_day` and `rebalance_satellite` methods and `burn_in` class
attribute are required in order for HistoricalSimulator to work properly. As
long as you include those, it's also possible to new Strategy classes using
your own trading logic.
'''

class SMAStrategy(HistoricalSimulator):
    '''
    Make sure *one* of the tickers has a 'track' key that equals True, as
    that will be asset whose simple moving average is used to make decisions
    for that the satellite in `self.on_new_day()`
    '''
    def __init__(self, Portfolio, burn_in=200, **kwargs):
        # determine number of burn in days needed -- e.g., for a 200-day simple
        # moving average based strategy, burn_in should be 200 days
        self.burn_in = burn_in # number of days for SMA

        #super(SMAStrategy, self).__init__(**kwargs)
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

        # indicators to track daily
        self.vol_streak = 0 # how many consecutive days with low volatility?
        self.vol_threshold = 1.01 # minimum safe multiple of SPY for entering
        self.can_enter = True # can we enter the market?
        self.retreat_period = 60 # minimum days to remain out after retreating
        self.days_out = 0 # how many consecutive days have you been in retreat?

    def on_new_day(self, ind_all, ind_active):
        '''
        FOR SIMPLE MOVING AVERAGE CASE ONLY
        (track SMA-related attributes)
        '''
        tracked_price = self.assets[self.track_tick]['df']['adjClose'][ind_all]
        tracked_sma = self.smas[self.track_tick][ind_active]

        # check if SMA is over our threshold and adjust streak counters
        if tracked_price >= tracked_sma * self.vol_threshold: # streak builds
            self.vol_streak += 1
            if self.vol_streak >= 3 and self.days_out >= self.retreat_period:
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
            self.vol_streak = 0

    def calc_mv_avg(self, prices, burn_in):
        '''
        SHOULD THIS HAVE ARGUMENTS OR BE A "PRIVATE" METHOD?

        Called from _calc_rolling_smas(). A faster method of calculating moving
        averages than slicing + np.mean. Takes a column of a pandas DataFrame.

        This method is also about an order of magnitude faster than
        prices.rolling(window=burn_in).mean()[burn_in-1:].values.

        Numerical precision apparently breaks down when len(prices) > 1e7 (so
        no worries in this scenario).

        Found here, along with concerns: https://stackoverflow.com/a/27681394
        Simplified here: https://stackoverflow.com/a/43286184
        '''
        #cumsum = np.cumsum(np.insert(prices.values, 0, 0))
        cumsum = np.cumsum(prices.values)
        #return (cumsum[burn_in:] - cumsum[:-burn_in]) / float(burn_in)
        cumsum[burn_in:] = cumsum[burn_in:] - cumsum[:-burn_in]
        return cumsum[burn_in - 1:] / float(burn_in)

    def _calc_rolling_smas(self):
        '''
        Called from __init__() of SMAStrategy.

        Calculate rolling simple moving average of closing price for each
        ticker. The length of a period is `burn_in`.

        Returns a dictionary with the same keys as `self.assets`. Each key
        contains an array of rolling simple moving avereages whose indices
        match up with `self.active_dates`.
        '''
        smas = {}
        for nm in self.assets.keys():
            prices = self.assets[nm]['df']['adjClose']

            # get array of all `burn_in`-day simple moving averages
            smas[nm] = self.calc_mv_avg(prices, self.burn_in)

        return smas

    def rebalance_satellite(self, ind_all, ind_active, curr_rb_ind,
                            verbose=False):
        '''
        FOR SIMPLE MOVING AVERAGE CASE ONLY, satellite rebalance procedure
        (after start date, move in/out of market depending on SPY price/SMA)

        Moves satellite entirely in or out of the market depending on simple
        moving average-related indicators. On satellite-only rebalances,
        decides whether to go in or out and completes any needed transactions
        in-method. When the whole portfolio is rebalanced, the decision of
        where to go is made in-method, but the (re-weighted) number of shares is
        returned for later use.
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)
        my_pr(f"it's a satellite; sat_only is {self.sat_only[curr_rb_ind]}; "
              f"${self.cash:.2f} in account")

        # exit if there are no satellite assets
        if len(self.sat_names) == 0:
            # empty list needed when called from self.rebalance_portfolio()
            return []

        # (purchases will be made using TODAY's PRICES, NOT yesterday's closes)
        in_mkt_tick = self.sat_names[0]
        in_mkt_pr = self.assets[in_mkt_tick]['df']['adjOpen'][ind_all]
        in_mkt_sh = self.assets[in_mkt_tick]['shares']

        out_mkt_tick = self.sat_names[-1]
        out_mkt_pr = self.assets[out_mkt_tick]['df']['adjOpen'][ind_all]
        out_mkt_sh = self.assets[out_mkt_tick]['shares']

        total_mkt_val = self.portfolio_value(ind_all, rebalance=True)

        # with enough consecutive days above SMA (and a buffer, if necessary)...
        if self.can_enter:
            # switch to in-market asset if not already there
            if out_mkt_sh > 0:
                # liquidate out_mkt holdings
                out_mkt_delta = -out_mkt_sh
                out_mkt_cash = out_mkt_sh * out_mkt_pr

                if self.sat_only[curr_rb_ind]: # use all available $$ to buy in
                    my_pr('1: liq out, buy in')
                    in_mkt_delta = int((self.cash + out_mkt_cash) // in_mkt_pr)
                else: # if total rebalance, re-weight and return shares to buy
                    my_pr('2: liq out, balance in')
                    in_mkt_delta = (total_mkt_val * self.sat_frac) // in_mkt_pr
                    return [in_mkt_delta, out_mkt_delta]
            # if already invested in in-market asset...
            else:
                if self.sat_only[curr_rb_ind]: # already in, so nothing to buy
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

                if self.sat_only[curr_rb_ind]: # use all available $$ to retreat
                    my_pr('5: liq in, buy out')
                    out_mkt_delta = int((self.cash + in_mkt_cash) // out_mkt_pr)
                else: # if total rebalance, find and return net share change
                    my_pr('6: liq in, balance out')
                    out_mkt_delta = (total_mkt_val*self.sat_frac) // out_mkt_pr
                    return [in_mkt_delta, out_mkt_delta]
            # if already invested in out-market asset...
            else:
                if self.sat_only[curr_rb_ind]: # already out, so nothing to buy
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
        self._make_rb_trades(names, deltas, ind_all, verbose=verbose)

class VolTargetStrategy(HistoricalSimulator):
    def __init__(self, Portfolio, burn_in=30, vol_target=.15, **kwargs):
        # determine number of burn in days needed -- e.g., for a 200-day simple
        # moving average based strategy, burn_in should be 200 days
        self.burn_in = burn_in # volatility period

        #super(VolTargetStrategy, self).__init__(**kwargs)
        super().__init__(Portfolio, **kwargs)

        # then, set strategy-specific attributes
        # set target `burn_in`-day annualized volatility for satellite portfolio
        self.vol_target = vol_target

        # calculate assets' rolling standard deviations from daily closes
        self.stds = self._calc_rolling_stds()

        # calculate inter-asset correlations
        self.corrs = self._calc_correlations()

    def on_new_day(self, ind_all, ind_active):
        '''
        FOR VOLATILITY TARGETING CASE ONLY
        ...anything to do on a daily basis?
        '''
        pass

    def _calc_rolling_stds(self):
        '''
        Called from __init__() of VolTargetStrategy.

        Calculate rolling standard deviation for each ticker. The length of a
        rolling period is `burn_in`.

        Returns a dictionary with the same keys as `self.assets`. Each key
        contains an array of rolling standard deviations whose indices match up
        with `self.active_dates`.
        '''
        stds = {}
        for nm in self.assets.keys():
            prices = self.assets[nm]['df']['adjClose']

            # record asset's daily logartihmic returns
            #(np.log(prices).diff().rolling(21).std() * np.sqrt(252)).values
            log_ret = np.log(prices).diff().fillna(0) # (change 0th entry to 0)

            # collect rolling `burn_in`-day standard deviations; slice out nans
            devs = log_ret.rolling(self.burn_in).std()[self.burn_in - 1:]

            # save the array of standard deviations after annualizing them
            stds[nm] = devs.values * np.sqrt(252) #*100
            # tried a numpy-only solution but the speed gain was minimal:
            # https://stackoverflow.com/questions/43284304/

        return stds

    def _calc_correlations(self):
        '''
        Called from __init__() of VolTargetStrategy.

        Calculate rolling inter-asset correlations for all tickers in
        `self.assets`. Returns a DataFrame that takes two asset labels as
        indices and gives back an array of correlation values between those
        assets.

        For example, corrs['TICK1']['TICK2'] (provide the ticker strings
        yourself) gives an array of correlations. The indices of the array
        match up with the results of _get_moving_stats() & `self.active_dates`.

        CHANGE? At the moment, an asset's correlation with itself is NaN instead of an array of 1s. It should save a little time in the loop, but does it matter?
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

    def rebalance_satellite(self, ind_all, ind_active, curr_rb_ind,
                            verbose=False):
        '''
        FOR VOLATILITY TARGETING CASE ONLY, satellite rebalance procedure
        [OJO! volatility == standard deviation (or sigma) == sqrt(variance)]

        Adjusts the in and out of market portions of the satellite to target
        user's chosen volatility. On satellite-only rebalances, it completes
        the resulting transactions in-method. On occasions when the whole
        portfolio is rebalanced, it returns the appropriate number of shares
        to buy/sell for all satellite assets.
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)
        my_pr(f"it's a satellite; sat_only is {self.sat_only[curr_rb_ind]}; "
              f"${self.cash:.2f} in account")

        # exit if there are no satellite assets
        if len(self.sat_names) == 0:
            # empty list needed when called from self.rebalance_portfolio()
            return []

        # What are my current satellite holdings worth?
        # (purchases will be made using TODAY's PRICES, NOT yesterday's closes)
        in_mkt_tick = self.sat_names[0]
        in_mkt_pr = self.assets[in_mkt_tick]['df']['adjOpen'][ind_all]
        in_mkt_sh = int(self.assets[in_mkt_tick]['shares'])
        in_mkt_val = in_mkt_pr * in_mkt_sh

        out_mkt_tick = self.sat_names[-1]
        out_mkt_pr = self.assets[out_mkt_tick]['df']['adjOpen'][ind_all]
        out_mkt_sh = int(self.assets[out_mkt_tick]['shares'])
        out_mkt_val = out_mkt_pr * out_mkt_sh

        # What can i spend during this rebalance?
        total_mkt_val = self.portfolio_value(ind_all, rebalance=True)
        can_spend = ((in_mkt_val + out_mkt_val + self.cash)
                     if self.sat_only[curr_rb_ind]
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

        my_pr({f"{k[0]*1e2:.0f}%": k[1]
               for k in np.round(np.stack((fracs, vol_tests), axis=1), 2)})
        my_pr(f"vol target is {self.vol_target:.2f}. frac in: {new_frac_in:.2f}")
        my_pr(f"current shares: {self.assets[in_mkt_tick]['shares']} in, {self.assets[out_mkt_tick]['shares']} out")

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

        # Return share change values if this is an entire-portfolio rebalance
        if not self.sat_only[curr_rb_ind]:
            return deltas
        else: # If this is a satellite-only rebalance, make the trades
            names = np.array(self.sat_names)
            deltas = np.array(deltas)
            self._make_rb_trades(names, deltas, ind_all, verbose=verbose)
