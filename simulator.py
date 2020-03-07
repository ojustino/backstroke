from datetime import datetime, timedelta
from portfolio_maker import PortfolioMaker
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import time

MY_API_KEY = '901a2a03f9d57935c22df22ae5a5377cb8de6f22'

class HistoricalSimulator():
    # AbstractBaseClass? in order to impose that children
    # should have certain methods that are not defined here
    '''
    `Portfolio`, `start_date`, `end_date`, `sat_rb_freq`, `tot_rb_freq`, `cash`
    '''
    # earliest start dates: 1998-11-22, 2007-05-22, 2012-10-21
    def __init__(self, Portfolio,
                 start_date=datetime(1998, 11, 22),
                 end_date=datetime(1998, 11, 22) + timedelta(days=365.25*8),
                 sat_rb_freq=6, tot_rb_freq=1, cash=1e4):
        # make sure a PortfolioMaker object is present
        if not isinstance(Portfolio, PortfolioMaker):
            raise ValueError('The first argument of HistoricalSimulator() must '
                             'be a PortfolioMaker() instance.')

        # estimate period needed to warm up strategy's statistic(s) (converting
        # real days to approx. market days) and subtract result from start_date
        mkt_to_real_days = 365.25 / 252.75 # denominator is avg mkt days in year
        buffer_days = int(self.burn_in * mkt_to_real_days) + 5
        self.open_date = start_date - timedelta(buffer_days)

        # save dates over which analysis will take place
        self.start_date = start_date
        self.end_date = end_date

        # save the core and satellite fractions
        self.core_frac = np.round(Portfolio._get_label_weights('core').sum(),6)
        self.sat_frac = np.round(1 - self.core_frac, 6)

        # how much money do we have to invest?
        # (is a property so an error is thrown if it goes negative)
        self._cash = float(cash)

        # validate proposed asset dictionary, then add historical data to it
        self.assets = self._validate_assets_dict(Portfolio)

        # how often a year should we rebalance the satellite portion?
        # and, how often a year should we rebalance the whole portfolio?
        if sat_rb_freq < tot_rb_freq:
            raise ValueError('satellite rebalance frequency must be greater '
                             'than or equal to total rebalance frequency')
        if (  ((12 % sat_rb_freq != 0 or sat_rb_freq % 1 != 0)
               and sat_rb_freq != 365.25)
            or (12 % tot_rb_freq != 0 or tot_rb_freq % 1 != 0)  ):
            raise ValueError('Allowed rebalance frequencies are 1, 2, 3, 4, '
                             '6, 8, and 12 times a year. `sat_rb_freq` can '
                             'also be 365.25 for daily rebalances.')
        self.sat_rb_freq = sat_rb_freq
        self.tot_rb_freq = tot_rb_freq

        # make arrays of all dates in set and all *active* dates (w/o burn-in)
        self.all_dates, self.active_dates = self._get_date_arrays()

        # at which indices in the ticker DataFrames will rebalances occur?
        # and, are those are they for the satellite only or the whole portfolio?
        self.rb_indices, self.sat_only = self._calc_rebalance_info()

        # make a DataFrame to track portfolio value over time
        self.strategy_results = pd.DataFrame({'date': self.active_dates,
                                              'value': np.zeros(len(self.active_dates))})
        if self.sat_frac > 0:
            self.satellite_results = self.strategy_results.copy()
        if self.core_frac > 0:
            self.core_results = self.strategy_results.copy()

        self.cash_over_time = self.strategy_results.copy()

        # where in the time loop are we?
        self.today = self.open_date

        # save convenience lists of core and satellite asset names
        self.core_names = [key for key, info in self.assets.items()
                           if info['label'] == 'core']
        in_mkt_nm = [key for key, info in self.assets.items()
                     if info['label'] == 'satellite' and info['in_mkt']]
        out_mkt_nm = [key for key, info in self.assets.items()
                     if info['label'] == 'satellite' and not info['in_mkt']]
        self.sat_names = in_mkt_nm + out_mkt_nm
        # (for satellite assets, make sure the in-market asset comes first)

        # run the loop?

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, value):
        if value < 0:
            raise ValueError('You cannot spend more cash than you have.')
        self._cash = value

    def portfolio_value(self, rebalancing=False, ind_all=None):
        '''
        Return the value of all assets currently held in the portfolio,
        including cash.

        If `rebalancing` == False, asset prices use the current day's
        (`self.today`) closing price, and if `rebalancing` == True, then
        assets are valuated using the current day's opening price.

        `ind_all` is the index of each ticker's historical DataFrame from which
        to get the price data.

        MAY NEED TO ADD AN ARGUMENT FOR BENCHMARK AS WELL
        '''
        if self.today < self.start_date:
            return self.cash

        if not isinstance(rebalancing, bool):
            raise ValueError("'rebalancing' must be a bool.")

        # IF ind_all == None do something else... but what???

        # determine whether to use open or close prices for assets
        #assets = self.assets
        col = 'adjOpen' if rebalancing else 'adjClose'

        # multiply shares held of each ticker by their current prices
        holdings = np.sum([info['shares'] * info['df'][col][ind_all]
                           for info in self.assets.values()
                           if info['label'] in ['core', 'satellite']])

        return self.cash + holdings

    def call_tiingo(self, tick, open_date, end_date=datetime.now()):
        '''
        Called in _build_assets_dict() but can also be used independently.

        Retrieve historical price data for ticker `tick` from `open_date` to
        `end_date`, convert it to a DataFrame, then return it.
        '''
        open_date = open_date.strftime('%Y-%m-%d') # (e.g. '1998-07-11')
        end_date = end_date.strftime('%Y-%m-%d')
        print(f"{tick} from {open_date} to {end_date}...")

        headers = {'Content-Type': 'application/json'}
        params = {
            'startDate': open_date,
            'endDate': end_date,
            'format': 'json',
            'resampleFreq': 'daily',
            'token': MY_API_KEY,
        }

        url = 'https://api.tiingo.com/tiingo/daily/' + tick + '/prices'
        resp = requests.get(url, params=params, headers=headers)
        assert resp.status_code == 200, 'HTTP status code was not 200'

        df = pd.DataFrame.from_dict(resp.json())
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _verify_dates(self, tick_info):
        '''
        Check whether any assets will have missing data based on user's proposed
        start and end times for the simulation. If so, throw an error.
        '''
        # are all assets active by self.start_date?
        for i, dt in enumerate(tick_info['startDate']):
            dt = pd.to_datetime(dt)
            tk = tick_info.iloc[i]['ticker']
            if dt > self.start_date:
                dt_str = dt.strftime('%Y-%m-%d')
                sd_str = self.start_date.strftime('%Y-%m-%d')
                raise ValueError(f"{tk}'s start date of {dt_str} is later "
                                 f"than your chosen start date of {sd_str}. "
                                 'Try an earlier start date or choose a '
                                 ' different ticker.')

        # are all assets still active by self.end_date?
        for i, dt in enumerate(tick_info['endDate']):
            dt = pd.to_datetime(dt)
            tk = tick_info.iloc[i]['ticker']
            if dt < self.end_date:
                dt_str = dt.strftime('%Y-%m-%d')
                ed_str = self.end_date.strftime('%Y-%m-%d')
                raise ValueError(f"{tk}'s end date of {dt_str} is earlier "
                                 f"than your chosen end date of {ed_str}. "
                                 'Try an earlier end date or choose a '
                                 'different ticker.')

    def _validate_assets_dict(self, Portfolio):
        '''
        Called in __init__() of HistoricalSimulator.

        Verifies that tickers from the `assets` dict in the user's provided
        instance of the PortfolioMaker class have valid dates. If so, it
        retrieves historical price data from Tiingo for each.

        Then, adds 'df' and 'shares' keys to each `assets[ticker]` dict; their
        respective values are the returned DataFrame and the number of 'ticker'
        shares currently held (0 to start).
        '''
        # run Portfolio's own validation function to be thorough
        Portfolio.check_assets()

        # verify that all assets are present over the user's entire date range
        self._verify_dates(Portfolio.tick_info)

        # if those tests pass, fetch historical data from online for each asset
        assets = copy.deepcopy(Portfolio.assets)
        for tick, info in assets.items():
            # get daily open/close data from Tiingo
            df = self.call_tiingo(tick, self.open_date, self.end_date)

            # add the dataframe to the ticker's dictionary information
            info['df'] =  df

            # initialize this asset with zero shares
            info['shares'] = 0

        # ensure that each asset has the same number of dates
        assert len(np.unique([len(assets[nm]['df']['date'])
                              for nm in assets])) == 1, ('some ticker '
                                                         'DataFrames are '
                                                         'missing dates')

        return assets

    def _get_date_arrays(self):
        '''
        Called in __init__() of HistoricalSimulator. Traverses downloaded
        historical data and returns an array with all available dates
        (all_dates) and another with the burn-in dates removed (active_dates).

        Also changes self.start_date to the next market day if the user's
        original choice is absent from the data.
        '''
        # pick a ticker (shouldn't matter which; all should have same dates)
        nm = list(self.assets.keys())[0]

        # only consider dates in active period (i.e., remove burn-in dates)
        all_dates = self.assets[nm]['df']['date'].values

        # save an array with only dates in active period (from start_date on)
        active_dates = all_dates[self.burn_in:]
        real_start = pd.to_datetime(active_dates[0])

        # change start_date to match data if user-given start_date isn't in set
        if (real_start.strftime('%Y-%m-%d')
            != self.start_date.strftime('%Y-%m-%d')):
            self.start_date = real_start.to_pydatetime()

        return all_dates, active_dates

    def _append_or_assign(self, obj, ind, sat_only_is=None):
        '''
        Called in _calc_rebalance_info(). `rb_indices` and `sat_only` can start
        out as lists or arrays, This method handles that ambiguity by first
        trying to append argument `ind` to `obj`. This is the case where
        `rb_indices` and `sat_only` grow one item at a time.

        If append() is not an attribute of `obj`, we revert to flipping the
        value of index `ind` in `obj`, since `obj` must be an array instead.
        This is the case where `rb_indices` and `sat_only` are pre-filled and
        the latter has some indices that needed to be changed from True to
        False.
        '''
        which = sat_only_is is not None
        try:
            obj.append(ind if not which else sat_only_is)
        except AttributeError:
            if which: # if sat_only (no changes to array version of rb_indices)
                obj[ind] = sat_only_is

        return obj

    def _last_day_of_month(self, year, month):
        '''
        Called in _calc_rebalance_info(). Reliably calculate the date of the
        specified month's final market day.
        '''
        next_month = datetime(year, month, 28) + timedelta(days=4)
        return (next_month - timedelta(days=next_month.day)).isoformat()

    def _calc_rebalance_info(self):
        '''
        Called in __init__() of HistoricalSimulator.

        Uses satellite and total portfolio rebalance frequencies to get an
        array of the indices of the 'date' column in each ticker's DataFrame
        that will trigger rebalance events. Returns that with a same-size,
        associated array that is True when the rebalance is for the satellite
        portion only and false when it's time for a full portflio rebalance.

        Frequencies of once a month or less will always rebalance on the
        penultimate market day of a qualifying month to try and avoid whipsawing
        from larger investors doing their own rebalancing on the last or first
        market day of the month.
        '''
        # calculate the months in which to perform each type of rebalance
        all_months = np.arange(1, 13)

        # get total rebalance months, shifting list to include start month
        tot_mths = all_months[all_months % (12 / self.tot_rb_freq) == 0]
        tot_mths = (tot_mths + self.start_date.month) % 12
        tot_mths[tot_mths == 0] += 12 # or else december would be 0

        # choose satellite rebalance strategy based on frequency
        if self.sat_rb_freq <= 12: # if monthly or less freqent...
            # get satellite rebalance months, perform the same shift
            sat_mths = all_months[all_months % (12 / self.sat_rb_freq) == 0]
            sat_mths = (sat_mths + self.start_date.month) % 12
            sat_mths[sat_mths == 0] += 12 # or else december would be 0

            # give total rebalances priority over satellite-only
            sat_mths = sat_mths[~np.in1d(sat_mths, tot_mths)]
            print('sat', sat_mths, '\ntot', tot_mths)

            # create list of indices of active_dates where rebalances occur
            # and another specifying which type (satellite or total?)
            rb_indices = []
            sat_only = []

        else: # if daily... (only. later could include every 2nd/3rd day, etc.)
            # ...then every month has rebalance events
            sat_mths = all_months
            print('sat', sat_mths, '\ntot', tot_mths)

            # include every active_date as a possible rebalance date
            # (total rebalance days will be flipped to True in sat_only later)
            rb_indices = np.arange(len(self.active_dates))
            sat_only = np.ones(len(self.active_dates)).astype(bool)

        go = time.time()
        yr = self.start_date.year
        while yr <= self.end_date.year:
            # make array with all eligible months
            months = np.arange(1 if yr != self.start_date.year
                               else self.start_date.month,
                               13 if yr != self.end_date.year
                               else self.end_date.month)
            # don't add 1 to end_month in last year -- we don't want it
            # included because data may not get to its penultimate market day

            print(months, yr)
            # limit months to those cleared for rebalance events
            eligible = [mth for mth in months
                        if mth in tot_mths or mth in sat_mths]

            # check every month in the current year...
            for mth in eligible:
                # automatically make start_date a total rebalance event
                if (mth == self.start_date.month
                    and yr == self.start_date.year):
                    rb_indices = self._append_or_assign(rb_indices, 0)
                    sat_only = self._append_or_assign(sat_only, 0,
                                                      sat_only_is=False)
                # in subsequent months, get month's penultimate market day
                else:
                    last_day = np.datetime64(self._last_day_of_month(yr, mth))
                    penult = np.where(self.active_dates < last_day)[0][-1]
                    rb_indices = self._append_or_assign(rb_indices, penult)
                    # note type of rebalance that takes place this month
                    kind = True if mth not in tot_mths else False
                    sat_only = self._append_or_assign(sat_only, penult,
                                                      sat_only_is=kind)

            yr += 1

        print(f"{time.time() - go:.3f} s for rebalance info loop")

        # make arrays and shift rb_indices to account for burn-in days
        rb_indices = np.array(rb_indices) + self.burn_in
        sat_only = np.array(sat_only)

        return rb_indices, sat_only

    def rebalance_portfolio(self, ind_all, ind_active, curr_rb_ind):
        '''
        Called in `self.begin_time_loop()`.

        General method that performs a whole-portfolio rebalance by
        re-weighting core assets in-method and gets needed changes for
        satellite assets from rebalance_satellite(). Then, completes the
        transactions needed to restore balance.

        The hope is that this method can work with any strategy by outsourcing
        the procedures that differ in the individual rebalance_satellite()
        methods from various Strategy classes. This assumes that the
        target weights for the core will not change over time and that total
        rebalances should always try to bring the portfolio back to them.

        If that changes, perhaps add a rebalance_core() method?
        '''
        print(f"it's a total; sat_only is {self.sat_only[curr_rb_ind]}; "
              f"${self.cash:.2f} in account")

        # get share changes for core assets
        deltas = []
        for core in self.core_names:
            ideal_frac = self.assets[core]['fraction']
            ideal_holdings = self.portfolio_value(True, ind_all) * ideal_frac

            curr_price = self.assets[core]['df']['adjOpen'][ind_all]
            curr_held = self.assets[core]['shares'] * curr_price

            delta_shares = (ideal_holdings - curr_held) // curr_price
            deltas.append(delta_shares)

        # get satellite assets' net share changes from their dedicated method
        deltas.extend(self.rebalance_satellite(ind_all, ind_active,
                                               curr_rb_ind))
        deltas = np.array(deltas)
        print('deltas:', deltas)

        # once we have deltas for all assets, find which require sells/buys
        to_sell = np.where(deltas < 0)[0]
        to_buy = np.where(deltas > 0)[0] # no action needed when deltas == 0

        # gather all names and prices in arrays that correspond with delta
        all_names = np.array(self.core_names + self.sat_names)
        prices = np.array([self.assets[nm]['df']['adjOpen'][ind_all]
                           for nm in all_names])

        # first, sell symbols that are currently overweighted in portfolio
        for i, nm in enumerate(all_names[to_sell]):
            share_change = deltas[to_sell[i]] # this is negative, so...
            self.cash -= prices[to_sell[i]] * share_change # ...increases $$
            print(f"sold {abs(share_change):.0f} shares of {nm} "
                  f"@${prices[to_sell[i]]:.2f} | ${self.cash:.2f} in account")
            self.assets[nm]['shares'] += share_change # ...decreases shares

        # then, buy underweighted symbols
        for i, nm in enumerate(all_names[to_buy]):
            share_change = deltas[to_buy[i]]
            self.cash -= prices[to_buy[i]] * share_change
            print(f"bought {share_change:.0f} shares of {nm} "
                  f"@${prices[to_buy[i]]:.2f} | ${self.cash:.2f} in account")
            self.assets[nm]['shares'] += share_change

    def begin_time_loop(self):
        '''
        Called in __init__ of HistoricalSimulator or by user????

        Step through all avilable dates in the historical data set, tracking and
        rebalancing the portfolio along the way. Buy at open, track stats at
        close.
        '''
        go = time.time()
        curr_rb_ind = 0
        for i, today in enumerate(self.all_dates):
            self.today = pd.to_datetime(today)
            # "PRE-OPEN": update daily indicators based on YESTERDAY's CLOSES
            if i >= self.burn_in:
                self.on_new_day(i, i - self.burn_in)

            # AT OPEN: rebalance if needed
            if i == self.rb_indices[curr_rb_ind]:
                # make rebalance calculations based on YESTERDAY'S STATS,
                # which come at index [i - burn_in] of smas, stds, etc.

                try:
                    print('**** on', self.today.strftime('%Y-%m-%d'),
                          '\nvol streak: ', self.vol_streak,
                          'can enter:', self.can_enter,
                          'days out:', self.days_out)
                except AttributeError:
                    print('**** on', self.today.strftime('%Y-%m-%d'))

                if self.sat_only[curr_rb_ind] == True:
                    # rebalance satellite portion
                    self.rebalance_satellite(i, i - self.burn_in, curr_rb_ind)
                else:
                    # rebalance total portfolio
                    self.rebalance_portfolio(i, i - self.burn_in, curr_rb_ind)

                if curr_rb_ind < len(self.rb_indices) - 1:
                    curr_rb_ind += 1

            # AT CLOSE: track stats
            if i >= self.burn_in:
                # this is slow... any way to speed it up?
                # maybe add values to a 1-D array and then join it with
                # active_dates in DataFrame after the loop is complete???
                # also, look here: https://stackoverflow.com/a/45983830
                curr_pf_val = self.portfolio_value(False, i)
                self.strategy_results.loc[i-self.burn_in, 'value'] = curr_pf_val
                self.cash_over_time.loc[i-self.burn_in, 'value'] = self.cash

        print(f"{time.time() - go:.3f} s for time loop")
