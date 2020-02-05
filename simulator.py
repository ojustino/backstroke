from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import time

MY_API_KEY = '901a2a03f9d57935c22df22ae5a5377cb8de6f22'

class HistoricalSimulator():
    # could use an argument for a custom list of ticker symbols
    # earliest start dates: 1998-11-22, 2007-05-22, 2012-10-21
    def __init__(self, timing='old', start_date=datetime(1998, 11, 22),
                 end_date=datetime(1998, 11, 22) + timedelta(days=365.25*8),
                 sat_frac=.5, sat_rb_freq=6, tot_rb_freq=1,
                 cash=1e4):
        # which time period should we test?
        if timing not in ['old', 'mid', 'now']:
            raise ValueError("choices for 'timing' are 'old' (1998-), "
                             "'mid' (mid 2006-), and 'now' (2010-)")

        # estimate period needed to warm up strategy's statistic(s) (converting
        # real days to approx. market days) and subtract result from start_date
        mkt_to_real_days = 365.25 / 252.75 # denominator is avg mkt days in year
        self.buffer_days = int(self.burn_in * mkt_to_real_days) + 5
        self.open_date = start_date - timedelta(self.buffer_days)

        # when should we start gathering data? check if dates are available
        if timing == 'now':
            earliest = datetime(2012, 1, 1)
        if timing == 'mid':
            earliest = datetime(2006, 8, 1)
        if timing == 'old':
            earliest = datetime(1998, 2, 1)

        if self.open_date < earliest:
                raise ValueError(f"Data for the timing={timing} period only "
                                 f"exists from {earliest.strftime('%Y-%m-%d')} "
                                 "onward. Check your start date and burn-in.")
        self.timing = timing
        self.start_date = start_date
        self.end_date = end_date

        # what fraction of the portfolio should the satellite take up?
        if 0 <= sat_frac <= 1:
            self.sat_frac = float(sat_frac)
            self.core_frac = 1 - self.sat_frac
        else:
            raise ValueError("'sat_frac' must be between 0 and 1 (inclusive)")

        # how much money do we have to invest?
        # (we'll make it a property to warn if it goes negative -- it shouldn't)
        self._cash = float(cash)

        # build dictionary of ticker symbols and associated data
        self.assets = self._build_assets_dict(self.open_date, self.end_date)

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

        # calculate rolling versions of stats on closing price data
        # (for now, that's `burn_in`-day moving average and standard deviation)
        self.smas, self.stds = self._get_moving_stats()

        # calculate rolling inter-asset correlations
        self.corrs = self._calc_correlations()

        # portfolio value property and dataframe
        self.strategy_results = pd.DataFrame({'date': self.active_dates,
                                              'value': np.zeros(len(self.active_dates))})
        if self.sat_frac > 0:
            self.satellite_results = self.strategy_results.copy()
        if self.core_frac > 0:
            self.core_results = self.strategy_results.copy()

        self.cash_over_time = self.strategy_results.copy()

        # where in the time loop are we?
        self.today = self.open_date

        # run the loop?

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, value):
        if value < 0:
            raise ValueError('you cannot spend more cash than you have')
        self._cash = value

    def portfolio_value(self, rebalancing=False, ind_all=None):
        if self.today < self.start_date:
            return self.cash

        if not isinstance(rebalancing, bool):
            raise ValueError("'rebalancing' must be a bool.")

        # IF ind_all == None do something else... but what???

        assets = self.assets
        col = 'adjOpen' if rebalancing else 'adjClose'

        # multiply shares held of each ticker by their current prices
        holdings = np.sum([(assets[nm]['shares']*assets[nm]['df'][col][ind_all])
                           for nm in assets
                           if assets[nm]['frac'] or nm.find('mkt') >= 0])

        return self.cash + holdings

    def _build_assets_dict(self, open_date, end_date):
        '''
        (HOPEFULLY) Called in __init__() of HistoricalSimulator.

        Turns the list of ticker symbols, labels, and weights into a dictionary.
        Default keys are the tickers' categories (us/intl, large/small cap,
        in/out of market for satellite, etc.). Values are another dict with
        the ticker symbol, a DataFrame with the ticker's historical data, the
        ticker itself (as a string), and the number of shares currently held (0
        to start).
        '''
        ticker_symbols = self._get_ticker_symbols()

        assets = {}
        for info in ticker_symbols:
            tick, key, frac = info

            # get daily open/close data from Tiingo
            df = self.call_tiingo(tick, open_date, end_date)

            # add this symbol and its info to the assets dict
            mini_dict = {'tick': tick, 'frac': frac, 'df': df, 'shares': 0}
            assets[key] = mini_dict

        # ensure that each asset has the same number of dates
        assert len(np.unique([len(assets[nm]['df']['date'])
                              for nm in assets])) == 1, ('some ticker '
                                                         'DataFrames are '
                                                         'missing dates')
        return assets

    def _get_ticker_symbols(self):
        '''
        Called in _build_assets_dict(). CONSIDER adding a way for
        the user to provide their own list (in the right format).
        '''
        # today's team (should work from ~2010 onward)
        if self.timing == 'now':
            ticker_symbols = [
                # core options
                ('SCHG', 'us_lg', .5 * self.core_frac), # us, large-cap growth
                ('SCHM', 'us_md', .15 * self.core_frac), # us, mid-cap blend
                ('EFG', 'int_lg', .15 * self.core_frac), # intl, large-cap growth
                ('BIV', 'gv_bnd', .15 * self.core_frac), # us, highly rated interm. bonds
                ('LQD', 'cp_bnd', .05 * self.core_frac), # us, corp-grade bond

                # satellite options
                ('TQQQ', 'in_mkt', None), # in-mkt, leveraged 3x to NASDAQ
                #('QLD', 'in_mkt', None), # in-mkt, leveraged 2x to NASDAQ
                #('SSO', 'in_mkt', None), # in-mkt, leveraged 2x to S&P 500
                #('DXQLX', 'in_mkt', None), # in-mkt, leveraged 2x to S&P 500, rebalanced MONTHLY
                ('TLT', 'out_mkt', None), # out-mkt, long-term (20+ yr.) treasury bonds
                #('IEF', 'out_mkt', None), # out-mkt, interm. (7-10 yr.) treasury bonds
                #('TMF', 'out_mkt', None), # out-mkt, leveraged 2x to long treasuries
                #('DXKLX', 'out_mkt', None), # out-mkt, leveraged 2x to interm. treasuries, rebalanced MONTHLY

                # tracking options
                ('SPY', 'bench_index', None),
                ('AGG', 'bench_bond', None),
            ]

        # mid guard (from July 2006 on, captures '08 crisis)
        elif self.timing == 'mid':
            ticker_symbols = [
                # core options
                ('VUG', 'us_lg', .5 * self.core_frac), # us, large-cap growth
                ('VO', 'us_md', .15 * self.core_frac), # us, mid-cap blend
                ('EFG', 'int_lg', .15 * self.core_frac), # intl, large-cap growth
                ('VBIIX', 'gv_bnd', .15 * self.core_frac), # us, highly rated interm. bonds
                ('ISHIX', 'cp_bnd', .05 * self.core_frac), # us, corp-grade bond (dead ringer for LQD)

                # satellite options
                ('SSO', 'in_mkt', None), # in-mkt, leveraged 2x to S&P 500
                #('DXQLX', 'in_mkt', None), # in-mkt, leveraged 2x to S&P 500, rebalanced MONTHLY
                ('TLT', 'out_mkt', None), # out-mkt, long-term (20+ yr.) treasury bonds
                #('IEF', 'out_mkt', None), # out-mkt, interm. (7-10 yr.) treasury bonds
                #('DXKLX', 'out_mkt', None), # out-mkt, leveraged 2x to interm. treasuries, rebalanced MONTHLY

                # tracking options
                ('SPY', 'bench_index', None),
                ('AGG', 'bench_bond', None),
            ]

        # oldest guard (from Jan 1998 on, captures dot-com and '08)
        elif self.timing == 'old':
            ticker_symbols = [
                # core options
                ('VIGRX', 'us_lg', .5 * self.core_frac), # us, large-cap growth
                #('VMCIX', 'us_md', .15 * self.core_frac), # us, mid-cap blend (est. 5/1998)
                ('PEXMX', 'us_md', .15 * self.core_frac), # us, mid-cap blend (est. 1/1998)
                #('VGTSX', 'int_lg', .15 * self.core_frac), # intl, large-cap BLEND (performs like EFG, though)
                ('PRFEX', 'int_lg', .15 * self.core_frac), # intl, large-cap growth
                ('VBIIX', 'gv_bnd', .15 * self.core_frac), # us, highly rated interm. bonds
                ('ISHIX', 'cp_bnd', .05 * self.core_frac), # us, corp-grade bond (dead ringer for LQD)

                # satellite options
                ('ULPIX', 'in_mkt', None), # in-mkt, leveraged 2x to S&P 500
                ('VUSTX', 'out_mkt', None), # out-mkt, long-term us govt bonds

                # tracking options
                ('SPY', 'bench_index', None),
                ('VBMFX', 'bench_bond', None), # basically a mutual fund version of AGG
            ]

        # ensure that all portfolio allocations add up to 1 (100%)
        core_fracs = np.sum([tup[-1] for tup in ticker_symbols
                             if tup[-1]]).round(5)
        if core_fracs + self.sat_frac != 1:
            raise ValueError("portfolio allocation fractions must add up to 1. "
                             f"current value is {core_fracs + self.sat_frac}")

        return ticker_symbols

    def call_tiingo(self, tick, open_date, end_date=datetime.now()):
        '''
        Called in _build_assets_dict() but can also be used independently.
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
        assert(resp.status_code == 200)

        df = pd.DataFrame.from_dict(resp.json())
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _get_date_arrays(self):
        '''
        Called in __init__() of HistoricalSimulator. Traverses downloaded
        historical data and returns an array with all available dates
        (all_dates) and another with the burn-in dates removed (active_dates).

        Also changes self.start_date to the next market day if the user's
        original choice is absent from the data.
        '''
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

        SHOULD all_dates, active_dates, start_date BE ARGUMENTS? SEEMS LIKE IT
        MAKES MORE SENSE TO RETURN THEM FROM ANOTHER METHOD. then you could
        remove leading underscore and it could be free for use like
        call_tiingo().

        Uses satellite and total portfolio rebalance frequencies to get an
        array of the indices of the 'date' column in each ticker's DataFrame
        that will trigger rebalance events. Returns that with  a same-size,
        associated array that is True when the rebalance is for the satellite
        portion only and false when it's time for a full portflio rebalance.

        Frequencies of once a month or less will always rebalance on the
        penultimate market day of a qualifying month to try and avoid whipsawing
        from larger investors doing their own rebalancing on the last or first
        market day of the month.

        Also assigns the true start date (after accounting for burn-in after
        open_date) and convenience arrays all_dates (every market day of
        historical data in the set) and active_dates (all_dates with burn-in
        dates removed) to class attributes for later use.
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
                    #rb_indices.append(0)
                    rb_indices = self._append_or_assign(rb_indices, 0)
                    sat_only = self._append_or_assign(sat_only, 0,
                                                      sat_only_is=False)
                # in subsequent months, get month's penultimate market day
                else:
                    last_day = np.datetime64(self._last_day_of_month(yr,
                                                                     mth))
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

    def calc_mv_avg(self, prices, burn_in):
        '''
        SHOULD THIS HAVE ARGUMENTS OR BE A "PRIVATE" METHOD?

        Called from _get_moving_stats() A faster method of calculating moving
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

    def _get_moving_stats(self):
        '''
        Called from __init__() of HistoricalSimulator.

        Calculate relevant rolling stats for each ticker. The length of a
        rolling period is `burn_in`. Not all are used in every strategy, but
        for now we calculate all stats at once.

        (At the moment, this is just moving average and standard deviation.)

        Results are arrays in dictionaries with same keys as `self.assets`.
        The arrays' indices match up with `self.active_dates`.
        '''
        smas = {}
        stds = {}
        for nm in self.assets.keys():
            prices = self.assets[nm]['df']['adjClose']

            # get array of all `burn_in`-day simple moving averages
            smas[nm] = self.calc_mv_avg(prices, self.burn_in)

            # get array of all `burn_in`-day volatilities
            stds[nm] = prices.rolling(window=self.burn_in).std()[self.burn_in
                                                                 -1:].values
            # tried a numpy-only solution but the speed gain was minimal:
            # https://stackoverflow.com/questions/43284304/

        return smas, stds

    def _calc_correlations(self):
        '''
        Called from __init__() of HistoricalSimulator.

        Calculate rolling inter-asset correlations for all tickers in
        `self.assets`. Returns a DataFrame that takes two asset labels as
        indices and gives back an array of correlation values between those
        assets.

        For example, corrs['in_mkt']['out_mkt'] gives an array of correlations.
        The indices of the array match up with the results of
        _get_moving_stats() and `self.active_dates`.

        CHANGE? At the moment, an asset's correlation with itself is NaN instead of an array of 1s. It should save a little time in the loop, but does it matter?
        '''
        all_keys = [key for key in self.assets.keys()]
        rem_keys = all_keys.copy()
        corrs = pd.DataFrame(columns=all_keys, index=all_keys)

        for nm1 in all_keys:
            rem_keys.remove(nm1)
            for nm2 in rem_keys:
                p1 = self.assets[nm1]['df']['adjClose']
                p2 = self.assets[nm2]['df']['adjClose']

                corr = p1.rolling(window=self.burn_in).corr(p2)[self.burn_in
                                                                -1:].values
                # (correlation from Pearson product-moment corr. matrix)
                # np.corrcoef(in, out) & np.cov(in,out) / (stddev_1 * stddev_2)
                # give the same result (index [0][1]) when the stddevs' ddof = 1

                corrs[nm1][nm2] = corr
                corrs[nm2][nm1] = corr

        return corrs

    def rebalance_portfolio(self, ind_all, ind_active, curr_rb_ind):
        '''
        Called in time loop...

        General method that performs a whole-portfolio rebalance by
        re-weighting core assets in-method and gets needed changes for
        satellite assets from rebalance_satellite(). Then, completes the
        transactions needed to restore balance.

        The hope is that this method can work with any strategy by outsourcing
        the procedures that differ in the individual rebalance_satellite()
        methods from the different ___Strategy classes. This assumes that the
        target weights for the core will not change over time and we should
        always try to bring the portfolio back to that balance.

        If that changes, perhaps add a rebalance_core() method?
        '''
        print(f"it's a total; sat_only is {self.sat_only[curr_rb_ind]}; "
              f"${self.cash:.2f} in account")
        # get total value of portfolio and gather asset names
        # (e.g. 'us_lg' or in_mkt', NOT tickers -- change this??)
        core_names = [key for key, val in self.assets.items() if val['frac']]
        sat_names = [key for key in self.assets.keys() if key.find('mkt') > 0]
        sat_names.sort() # in market asset should come first
        deltas = []

        # get share changes for core assets
        for core in core_names:
            ideal_frac = self.assets[core]['frac']
            ideal_holdings = self.portfolio_value(True, ind_all) * ideal_frac

            curr_price = self.assets[core]['df']['adjOpen'][ind_all]
            curr_held = self.assets[core]['shares'] * curr_price

            delta_shares = (ideal_holdings - curr_held) // curr_price
    #         delta_holdings = ((delta_shares if delta_shares >= 0
    #                            else delta_shares - 1) * curr_price)
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
        all_names = np.array(core_names + sat_names)
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
        rebalancing the portfolio along the way.
        '''
        go = time.time()
        curr_rb_ind = 0
        for i, today in enumerate(self.all_dates):
            self.today = pd.to_datetime(today)
            # "PRE-OPEN": update daily indicators with info up to YESTERDAY's CLOSES
            if i >= self.burn_in:
                self.on_new_day(i, i - self.burn_in)

            # AT OPEN: rebalance if needed
            if i == self.rb_indices[curr_rb_ind]:
                # make calculations based on PREVIOUS DAY'S STATS,
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
