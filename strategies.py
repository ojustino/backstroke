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

    def rebalance_satellite(self, ind_all, ind_active, curr_rb_ind):
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
        # (purchases will be made using TODAY's PRICES, NOT yesterday's closes)
        in_mkt_tick = self.sat_names[0]
        in_mkt_pr = self.assets[in_mkt_tick]['df']['adjOpen'][ind_all]
        in_mkt_sh = self.assets[in_mkt_tick]['shares']

        out_mkt_tick = self.sat_names[-1]
        out_mkt_pr = self.assets[out_mkt_tick]['df']['adjOpen'][ind_all]
        out_mkt_sh = self.assets[out_mkt_tick]['shares']

        total_mkt_val = self.portfolio_value(True, ind_all)

        print(f"it's a satellite; sat_only is {self.sat_only[curr_rb_ind]}; "
              f"${self.cash:.2f} in account")
        # with enough consecutive days above SMA (and a buffer, if necessary)...
        if self.can_enter:# and self.active:
            # switch to in-market asset if not already there
            if out_mkt_sh > 0:
                # liquidate out_mkt holdings
                self.cash += out_mkt_pr * out_mkt_sh
                self.assets[out_mkt_tick]['shares'] = 0
                print(f"sold {out_mkt_sh:.0f} shares of {out_mkt_tick} "
                      f"@{out_mkt_pr:.2f} | ${self.cash:.2f} in account")

                if self.sat_only[curr_rb_ind]: # use all available $$ to buy in
                    print('1: liq out, buy in')
                    in_mkt_sh = self.cash // in_mkt_pr
                    self.assets[in_mkt_tick]['shares'] = in_mkt_sh
                    self.cash -= in_mkt_pr * in_mkt_sh
                    print(f"bought {in_mkt_sh:.0f} shares of {in_mkt_tick} "
                          f"@{in_mkt_pr:.2f} | ${self.cash:.2f} in account")
                else: # if total rebalance, re-weight and return shares to buy
                    print('2: liq out, balance in')
                    delta = (total_mkt_val * self.sat_frac) // in_mkt_pr
                    return [delta, 0]
            # if already invested in in-market asset...
            else:
                if self.sat_only[curr_rb_ind]: # already in, so nothing to buy
                    print('3: already in, remain in')#pass
                else: # if total rebalance, find and return net share change
                    print('4: already in, balance in')
                    in_hold_val = in_mkt_sh * in_mkt_pr
                    delta = (total_mkt_val * self.sat_frac
                             - in_hold_val) // in_mkt_pr
                    return [delta, 0]
        # without enough consecutive days above SMA or enough buffer...
        elif not self.can_enter:# and self.active:
            # retreat to out of market asset if not already there
            if in_mkt_sh > 0:
                # liquidate in_mkt holdings
                self.cash += in_mkt_pr * in_mkt_sh
                self.assets[in_mkt_tick]['shares'] = 0
                print(f"sold {in_mkt_sh:.0f} shares of {in_mkt_tick} "
                      f"@{in_mkt_pr:.2f} | ${self.cash:.2f} in account")

                if self.sat_only[curr_rb_ind]: # use all available $$ to retreat
                    print('5: liq in, buy out')
                    out_mkt_sh = self.cash // out_mkt_pr
                    self.assets[out_mkt_tick]['shares'] = out_mkt_sh
                    self.cash -= out_mkt_pr * out_mkt_sh
                    print(f"bought {out_mkt_sh:.0f} shares of {out_mkt_tick} "
                          f"@{out_mkt_pr:.2f} | ${self.cash:.2f} in account")
                else: # if total rebalance, find and return net share change
                    print('6: liq in, balance out')
                    delta = (total_mkt_val * self.sat_frac) // out_mkt_pr
                    return [0, delta]
            # if already invested in out-market asset...
            else:
                if self.sat_only[curr_rb_ind]: # already out, so nothing to buy
                    print('7: already out, remain out')#pass
                else: # if total rebalance, find and return net share change
                    print('8: already out, balance out')
                    out_hold_val = out_mkt_sh * out_mkt_pr
                    delta = (total_mkt_val * self.sat_frac
                             - out_hold_val) // out_mkt_pr
                    return [0, delta]

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

    def rebalance_satellite(self, ind_all, ind_active, curr_rb_ind):
        '''
        FOR VOLATILITY TARGETING CASE ONLY, satellite rebalance procedure
        [OJO! volatility == standard deviation (or sigma) == sqrt(variance)]

        Adjusts the in and out of market portions of the satellite to target
        user's chosen volatility. On satellite-only rebalances, it completes
        the resulting transactions in-method. On occasions when the whole
        portfolio is rebalanced, it returns the appropriate number of shares
        to buy/sell for all satellite assets.
        '''

        print(f"it's a satellite; sat_only is {self.sat_only[curr_rb_ind]}; "
              f"${self.cash:.2f} in account")
        # What are my current satellite holdings worth?
        # (purchases will be made using TODAY's PRICES, NOT yesterday's closes)
        in_mkt_tick = self.sat_names[0]
        in_mkt_pr = self.assets[in_mkt_tick]['df']['adjOpen'][ind_all]
        in_mkt_sh = self.assets[in_mkt_tick]['shares']
        in_mkt_val = in_mkt_pr * in_mkt_sh

        out_mkt_tick = self.sat_names[-1]
        out_mkt_pr = self.assets[out_mkt_tick]['df']['adjOpen'][ind_all]
        out_mkt_sh = self.assets[out_mkt_tick]['shares']
        out_mkt_val = out_mkt_pr * out_mkt_sh

        # What can i spend during this rebalance?
        total_mkt_val = self.portfolio_value(True, ind_all)
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
        new_pct_in = fracs[np.argmin(np.abs(vol_tests - self.vol_target))]
        new_pct_out = 1 - new_pct_in

        print([k.tolist()
               for k in np.round(np.stack((fracs, vol_tests), axis=1), 2)])
        print(f"vol target is {self.vol_target:.2f}. frac in: {new_pct_in:.2f}")

        # Find the resulting net change in shares held
        in_delta = (can_spend * new_pct_in - in_mkt_val) // in_mkt_pr
        out_delta = (can_spend * new_pct_out - out_mkt_val) // out_mkt_pr
        deltas = [in_delta, out_delta]

        # Return share delta values if this is an entire-portfolio rebalance
        if not self.sat_only[curr_rb_ind]:
            return deltas

        # Otherwise, find which satellite assets require sells or buys
        sat_names = np.array(self.sat_names)
        prices = np.array([in_mkt_pr, out_mkt_pr])
        deltas = np.array(deltas)
        to_sell = np.where(deltas < 0)[0]
        to_buy = np.where(deltas > 0)[0] # no action needed when deltas == 0

        # Then, execute the necessary trades to match the new weightings
        # (selling first hopefully leaves enough cash to buy afterward)
        for i, nm in enumerate(sat_names[to_sell]):
            share_change = deltas[to_sell[i]] # this is negative, so...
            self.cash -= prices[to_sell[i]] * share_change # ...increases $$
            print(f"sold {abs(share_change):.0f} shares of {nm} "
                  f"@${prices[to_sell[i]]:.2f} | ${self.cash:.2f} in account")
            self.assets[nm]['shares'] += share_change # ...decreases shares

        # (then, buy)
        for i, nm in enumerate(sat_names[to_buy]):
            share_change = deltas[to_buy[i]]
            self.cash -= prices[to_buy[i]] * share_change
            print(f"bought {share_change:.0f} shares of {nm} "
                  f"@${prices[to_buy[i]]:.2f} | ${self.cash:.2f} in account")
            self.assets[nm]['shares'] += share_change
