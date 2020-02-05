from simulator import HistoricalSimulator
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

class SMAStrategy(HistoricalSimulator):
    def __init__(self, burn_in=200, **kwargs):
        # determine number of burn in days needed; estimate how many real days that is
        # and add it to end date. necessary to do this before super() call
        self.burn_in = burn_in # number of days for SMA

        #super(SMAStrategy, self).__init__(**kwargs)
        super().__init__(**kwargs)

        # daily indicators for SIMPLE MOVING AVERAGE STRATEGY ONLY
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
        spy_price = self.assets['bench_index']['df']['adjClose'][ind_all]
        spy_sma = self.smas['bench_index'][ind_active]
        #print('SPY close:', f"{spy_price:.2f}", '200 day SMA:', f"{spy_sma:.2f}")
        #plt.scatter(all_dates[ind_all], spy_price, c='#ce1141', s=1)
        #plt.scatter(all_dates[ind_all], spy_sma, c='#13274f', s=1)

        # check if SMA is over our threshold and adjust streak counters
        if spy_price >= spy_sma * self.vol_threshold: # streak builds
            self.vol_streak += 1
            if self.vol_streak >= 3 and self.days_out >= self.retreat_period:
                # if we were out, get back in
                self.can_enter = True
            elif self.days_out < self.retreat_period: # if we're already in...
                self.days_out += 1
        else: # streak broken, get out of the market
            if self.can_enter == True: # if we were in...
                self.days_out = 0
            else: # if we were already out...
                self.days_out += 1
            self.can_enter = False
            self.vol_streak = 0

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
        # using TODAY's PRICES, NOT yesterday's closes
        in_mkt_pr = self.assets['in_mkt']['df']['adjOpen'][ind_all]
        in_mkt_sh = self.assets['in_mkt']['shares']
        out_mkt_pr = self.assets['out_mkt']['df']['adjOpen'][ind_all]
        out_mkt_sh = self.assets['out_mkt']['shares']
        total_mkt_val = self.portfolio_value(True, ind_all)

        print(f"it's a satellite; sat_only is {self.sat_only[curr_rb_ind]}; "
              f"${self.cash:.2f} in account")
        # with enough consecutive days above SMA (and a buffer, if necessary)...
        if self.can_enter:# and self.active:
            # switch to in-market asset if not already there
            if out_mkt_sh > 0:
                # liquidate out_mkt holdings
                self.cash += out_mkt_pr * out_mkt_sh
                self.assets['out_mkt']['shares'] = 0
                print(f"sold {out_mkt_sh:.0f} of out_mkt @{out_mkt_pr:.2f} | "
                      f"${self.cash:.2f} in account")

                if self.sat_only[curr_rb_ind]: # use all available $$ to buy in
                    print('1: liq out, buy in')
                    in_mkt_sh = self.cash // in_mkt_pr
                    self.assets['in_mkt']['shares'] = in_mkt_sh
                    self.cash -= in_mkt_pr * in_mkt_sh
                    print(f"bought {in_mkt_sh:.0f} of in_mkt @{in_mkt_pr:.2f} | "
                          f"${self.cash:.2f} in account")
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
                self.assets['in_mkt']['shares'] = 0
                print(f"sold {in_mkt_sh:.0f} of in_mkt @{in_mkt_pr:.2f} | "
                      f"${self.cash:.2f} in account")

                if self.sat_only[curr_rb_ind]: # use all available $$ to retreat
                    print('5: liq in, buy out')
                    out_mkt_sh = self.cash // out_mkt_pr
                    self.assets['out_mkt']['shares'] = out_mkt_sh
                    self.cash -= out_mkt_pr * out_mkt_sh
                    print(f"bought {out_mkt_sh:.0f} of out_mkt @{out_mkt_pr:.2f} | "
                          f"${self.cash:.2f} in account")
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
    def __init__(self, burn_in=30, **kwargs):
        # determine number of burn in days needed; estimate how many real days that is
        # and add it to end date. necessary to do this before super() call
        self.burn_in = burn_in # volatility period

        super(VolTargetStrategy, self).__init__(**kwargs)

        # daily indicator(s?) for VOLATILITY TARGETING STRATEGY ONLY
        self.vol_target = .15 # what volatility can we handle in the satellite?
        # does vol_target need to be multiplied by number of biz days in year?
        # i.e., are we working with ANNUALIZED volatility?

    def on_new_day(self, ind_all, ind_active):
        '''
        FOR VOLATILITY TARGETING CASE ONLY
        ...anything to do on a daily basis?
        '''
        pass

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
        # using TODAY's PRICES, NOT yesterday's closes
        in_mkt_pr = self.assets['in_mkt']['df']['adjOpen'][ind_all]
        in_mkt_sh = self.assets['in_mkt']['shares']
        in_mkt_val = in_mkt_pr * in_mkt_sh

        out_mkt_pr = self.assets['out_mkt']['df']['adjOpen'][ind_all]
        out_mkt_sh = self.assets['out_mkt']['shares']
        out_mkt_val = out_mkt_pr * out_mkt_sh

        # What can i spend during this rebalance?
        total_mkt_val = self.portfolio_value(True, ind_all)
        can_spend = ((in_mkt_val + out_mkt_val + self.cash)
                     if self.sat_only[curr_rb_ind]
                     else total_mkt_val * self.sat_frac)
        # (available sum to spend depends on rebalance type)

        # Retrieve `burn_in`-day standard deviation and correlation values
        stddev_in = self.stds['in_mkt'][ind_active]
        stddev_out = self.stds['out_mkt'][ind_active]
        correlation = self.corrs['in_mkt']['out_mkt'][ind_active]

        # Test different in/out weights and record resulting volatilites
        n_tests = 21
        vol_tests = np.zeros(n_tests)
        fracs = np.linspace(1, 0, n_tests, endpoint=True)
        for i, frac_in in enumerate(fracs):
            frac_out = 1 - frac_in

            # calc two asset standard deviation
            # (gives a decimal, not a percentage!)
            curr_vol = np.sqrt((frac_in**2 * stddev_in**2)
                               + (frac_out**2 * stddev_out**2)
                               + (2 * frac_in * frac_out
                                  * stddev_in * stddev_out * correlation))

            # ensure this index isn't chosen if curr_vol == 0
            vol_tests[i] = curr_vol if curr_vol > 0 else 1e2

        # Isolate the weighting that comes closet to our target volatility
        # (ties will go to the combination with a higher in_mkt fraction)
        new_pct_in = fracs[np.argmin(np.abs(vol_tests - self.vol_target))]
        new_pct_out = 1 - new_pct_in

        print([k.tolist() for k in np.round(np.stack((fracs, vol_tests), axis=1), 2)])
        print(f"vol target is {self.vol_target:.2f}; frac in = {new_pct_in:.2f}")

        # find the resulting net change in shares held
        in_delta = (can_spend * new_pct_in - in_mkt_val) // in_mkt_pr
        out_delta = (can_spend * new_pct_out - out_mkt_val) // out_mkt_pr
        deltas = [in_delta, out_delta]

        # Return share delta values if this is an entire-portfolio rebalance
        if not self.sat_only[curr_rb_ind]:
            return deltas

        # If satellite-only, find which assets require sells or buys
        all_names = np.array(['in_mkt', 'out_mkt'])
        prices = np.array([in_mkt_pr, out_mkt_pr])
        deltas = np.array(deltas)
        to_sell = np.where(deltas < 0)[0]
        to_buy = np.where(deltas > 0)[0] # no action needed when deltas == 0

        # Then, execute the necessary trades to match the new weightings
        # (selling first hopefully leaves enough cash to buy afterward)
        for i, nm in enumerate(all_names[to_sell]): # this is negative, so...
            share_change = deltas[to_sell[i]]
            self.cash -= prices[to_sell[i]] * share_change # ...increases $$
            print(f"sold {abs(share_change):.0f} shares of {nm} "
                  f"@${prices[to_sell[i]]:.2f} | ${self.cash:.2f} in account")
            self.assets[nm]['shares'] += share_change # ...decreases shares

        # (then, buy)
        for i, nm in enumerate(all_names[to_buy]):
            share_change = deltas[to_buy[i]]
            self.cash -= prices[to_buy[i]] * share_change
            print(f"bought {share_change:.0f} shares of {nm} "
                  f"@${prices[to_buy[i]]:.2f} | ${self.cash:.2f} in account")
            self.assets[nm]['shares'] += share_change
