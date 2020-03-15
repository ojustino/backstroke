#!/usr/bin/python3
from datetime import datetime, timedelta
from portfolio_maker import PortfolioMaker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests
import time

MY_API_KEY = '901a2a03f9d57935c22df22ae5a5377cb8de6f22'

class HistoricalSimulator():
    # AbstractBaseClass? in order to impose that children
    # should have certain methods that are not defined here
    '''
    `Portfolio`, `start_date`, `end_date`, `sat_rb_freq`, `tot_rb_freq`,
    `reinvest_dividends`, and `cash`
    '''
    # earliest start dates: 1998-11-22, 2007-05-22, 2012-10-21
    def __init__(self, Portfolio,
                 start_date=datetime(1998, 11, 22),
                 end_date=datetime(1998, 11, 22) + timedelta(days=365.25*8),
                 sat_rb_freq=6, tot_rb_freq=1, reinvest_dividends=False,
                 cash=1e4):
        # make sure a PortfolioMaker object is present
        if not isinstance(Portfolio, PortfolioMaker):
            raise ValueError('The first argument of HistoricalSimulator() must '
                             'be a PortfolioMaker() instance.')

        # how often a year should we rebalance the satellite portion?
        # and, how often a year should we rebalance the whole portfolio?
        if (  ((12 % sat_rb_freq != 0 or sat_rb_freq % 1 != 0)
               and sat_rb_freq != 365.25)
            or (12 % tot_rb_freq != 0 or tot_rb_freq % 1 != 0)  ):
            raise ValueError('Allowed rebalance frequencies are 1, 2, 3, 4, '
                             '6, 8, and 12 times a year. `sat_rb_freq` can '
                             'also be 365.25 for daily rebalances.')
        if sat_rb_freq < tot_rb_freq:
            raise ValueError('satellite rebalance frequency must be greater '
                             'than or equal to total rebalance frequency')
        self.sat_rb_freq = sat_rb_freq
        self.tot_rb_freq = tot_rb_freq

        # estimate period needed to warm up strategy's statistic(s) (converting
        # real days to approx. market days) and subtract result from start_date
        mkt_to_real_days = 365.25 / 252.75 # denominator is avg mkt days in year
        buffer_days = int(self.burn_in * mkt_to_real_days) + 5
        self.open_date = start_date - timedelta(buffer_days)

        # save dates over which analysis will take place
        self.start_date = start_date
        self.end_date = end_date

        # validate proposed asset dictionary, then add historical data to it
        self.assets = self._validate_assets_dict(Portfolio)

        # make arrays of all dates in set and all *active* dates (sans burn-in)
        self.all_dates, self.active_dates = self._get_date_arrays()

        # at which indices in the ticker DataFrames will rebalances occur?
        # and, are they for the satellite only or the whole portfolio?
        self.rb_indices, self.sat_only = self._calc_rebalance_info()

        # save preference for handling dividend payouts
        self.reinvest_dividends = reinvest_dividends

        # track remaining money in main and benchmark portfolios
        # (are properties, so an error is thrown if they go negative)
        # (go Decimal here?)
        self._cash = float(cash)
        self._bench_cash = self._cash
        self._starting_cash = self._cash

        # save the core and satellite fractions
        self.core_frac = np.round(Portfolio._get_label_weights('core').sum(),6)
        self.sat_frac = np.round(1 - self.core_frac, 6)

        # make DataFrames to track main and benchmark portfolio values over time
        self.strategy_results = pd.DataFrame({
            'date': self.active_dates,
            'value': np.zeros(len(self.active_dates))})
        self.bench_results = self.strategy_results.copy()

        # make DataFrames to track portfolios that are 100% core and 100% sat
        if self.sat_frac > 0:
            self.satellite_results = self.strategy_results.copy()
        if self.core_frac > 0:
            self.core_results = self.strategy_results.copy()

        # make DataFrame to track free cash in main portfolio over time
        self.cash_over_time = self.strategy_results.copy()

        # track the current simulation date
        self.today = self.open_date

        # save convenience lists of core, satellite, and benchmark asset names
        self.core_names = [key for key, info in self.assets.items()
                           if info['label'] == 'core']

        in_mkt_nm = [key for key, info in self.assets.items()
                     if info['label'] == 'satellite' and info['in_mkt']]
        out_mkt_nm = [key for key, info in self.assets.items()
                     if info['label'] == 'satellite' and not info['in_mkt']]
        self.sat_names = in_mkt_nm + out_mkt_nm
        # (for satellite assets, make sure the in-market asset comes first)

        self.bench_names = [key for key, info in self.assets.items()
                            if info['label'] == 'benchmark']

        # run the loop?

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, value):
        if value < 0:
            raise ValueError('More cash was spent than remains'
                             ' in main portfolio.')
        self._cash = value

    @property
    def bench_cash(self):
        return self._bench_cash

    @bench_cash.setter
    def bench_cash(self, value):
        if value < 0:
            raise ValueError('More cash was spent than remains '
                             'in benchmark portfolio.')
        self._bench_cash = value

    def portfolio_value(self, ind_all=None,
                        main_portfolio=True, rebalance=False):
        '''
        Return the value of all assets currently held in the portfolio,
        including cash.

        `ind_all` is the integer index of each ticker's historical DataFrame
        from which to get the price data. It aligns with self.all_dates.

        If `main_portfolio` == True (default), the method returns the value of
        main strategy's core/satellite portfolio. If `main_portfolio` == False,
        the method returns the value of the benchmark portfolio.

        If `rebalance` == False (default), asset prices use the current day's
        (`self.today`) closing price, and if `rebalance` == True, then assets
        are valuated using the current day's opening price.
        '''
        if not isinstance(rebalance, bool):
            raise ValueError("'rebalance' must be a bool.")

        # get remaining cash for the chosen portfolio
        cash = self.cash if main_portfolio else self.bench_cash

        # if current sim hasn't reached the first trading date, return cash only
        if self.today < self.start_date:
            return cash

        # IF ind_all == None do something else... but what???

        # determine whether to use open or close prices for assets
        col = 'adjOpen' if rebalance else 'adjClose'

        # collect labels for assets in the chosen portfolio
        labels = {'core', 'satellite'} if main_portfolio else {'benchmark'}

        # multiply shares held of each ticker by their current prices
        holdings = np.sum([info['shares'] * info['df'][col][ind_all]
                           for info in self.assets.values()
                           if info['label'] in labels])

        return cash + holdings

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
                                 'different ticker.')

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
        # add a standard benchmark portfolio if one wasn't provided
        if len([tk for tk, info in Portfolio.assets.items()
                if info['label'] == 'benchmark']) == 0:
            # ensure that the potential additions exist during the simulation
            # period and are not already listed as core/satellite assets
            # (a future structural change might allow multiply-labeled assets)
            bench_stock = bench_bond = None

            # the stock index will be an ETF/mutual fund tracking the S&P 500
            if (  self.open_date >= datetime(1993, 1, 29)
                  and 'SPY' not in Portfolio.assets.keys()  ):
                bench_stock = 'SPY'
            elif (  self.open_date >= datetime(1976, 8, 31)
                    and 'VFINX' not in Portfolio.assets.keys()  ):
                bench_stock = 'VFINX'

            # the bond index will be an ETF/mutual fund tracking the
            # Barclays US Aggregate Bonds Index
            if (  self.open_date >= datetime(2003, 12, 31)
                  and 'AGG' not in Portfolio.assets.keys()  ):
                bench_bond = 'AGG'
            elif (  self.open_date >= datetime(1986, 12, 31)
                    and 'VBMFX' not in Portfolio.assets.keys()  ):
                bench_bond = 'VBMFX'

            # if valid stock & bond tickers were found, add the portfolio
            # (following the popular 60% stock/40% bond allocation model)
            if bench_stock is not None and bench_bond is not None:
                Portfolio.add_ticker(bench_stock, .6, label='benchmark')
                Portfolio.add_ticker(bench_bond, .4, label='benchmark')
            # else, the benchmark portfolio remains empty

        # run Portfolio's own validation function to be thorough
        Portfolio.check_assets()

        # verify that all assets are present over the user's entire date range
        self._verify_dates(Portfolio.tick_info)

        # if those tests pass, fetch historical data from online for each asset
        assets = pickle.loads(pickle.dumps(Portfolio.assets, -1))
        # (faster than copy.deepcopy for this use case)

        for tick, info in assets.items():
            # get daily open/close data from Tiingo
            df = self.call_tiingo(tick, self.open_date, self.end_date)

            # add the dataframe to the ticker's dictionary information
            info['df'] =  df

            # initialize this asset with zero shares
            # (go Decimal here?)
            info['shares'] = 0

        # ensure that each asset has the same number of dates
        num_dates = np.unique([len(assets[nm]['df']['date']) for nm in assets])
        assert len(num_dates) == 1, 'some ticker DataFrames are missing dates'

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
        out as lists or arrays. This method handles that ambiguity by first
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
        tot_mths[tot_mths == 0] = 12 # or else december would be 0

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

        else: # if daily... (only. in future could do every 2, 3 days and so on)
            # ...then every month has rebalance events
            sat_mths = all_months
            print('sat', sat_mths, '\ntot', tot_mths)

            # include every active_date as a possible rebalance date
            # (total rebalance days will be flipped to True in sat_only later)
            rb_indices = np.arange(len(self.active_dates))
            sat_only = np.ones(len(self.active_dates)).astype(bool)

        #go = time.time()
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
        #print(f"{time.time() - go:.3f} s for rebalance info loop")

        # make arrays and shift rb_indices to account for burn-in days
        rb_indices = np.array(rb_indices) + self.burn_in
        sat_only = np.array(sat_only)

        return rb_indices, sat_only

    def _get_static_rb_changes(self, names, ind_all, main_portfolio=True):
        '''
        Called in `rebalance_portfolio()`.

        Returns an array with the changes in shares for the tickers in `names`
        needed to rebalance the portfolio in question. The method name refers
        to "static" assets because it only works for assets whose target
        portfolio percentage does not change -- core or benchmark.

        `ind_all` is the integer index of each ticker's historical DataFrame
        from which to get the price data. It aligns with self.all_dates.

        `names` is a list of assets who share the same label; it should
        typically either be `self.core_names` or `self.bench_names`.

        `main_portfolio` decides whether to find changes for the main
        strategy's core/satellite portfolio (True) or the benchmark portfolio
        (False).
        '''
        # get total value for portfolio in question
        pf_value = self.portfolio_value(ind_all, main_portfolio=main_portfolio,
                                        rebalance=True)

        # get share changes for assets in `names`
        deltas = []
        for name in names:
            ideal_frac = self.assets[name]['fraction']
            ideal_holdings = pf_value * ideal_frac

            curr_price = self.assets[name]['df']['adjOpen'][ind_all]
            curr_held = self.assets[name]['shares'] * curr_price

            delta_shares = (ideal_holdings - curr_held) // curr_price
            deltas.append(delta_shares)

        return deltas

    def _make_rb_trades(self, names, deltas, ind_all, main_portfolio=True,
                        verbose=False):
        '''
        Called in `rebalance_portfolio()` or the child Strategy class'
        `rebalance_satellite()`.

        Completes the transactions needed to rebalance a portfolio.

        `names` is an array of assets to rebalance. It should typically be
        `np.array(self.core_names + self.sat_names)` (if called from
        rebalance_portfolio()), `np.array(self.sat_names)` alone (if called from
        rebalance_satellite()), or `np.array(self.bench_names)` (if called from
        rebalance_portfolio() for the benchmark portfolio).

        `deltas` is an array with the corresponding share changes for the
        assets in `names`.

        `ind_all` is the integer index of each ticker's historical DataFrame
        from which to get the price data. It aligns with self.all_dates.

        `main_portfolio` decides whether to find changes for the main
        strategy's core/satellite portfolio (True) or the benchmark portfolio
        (False).

        `verbose` is a boolean that controls whether or not to print
        information on completed trades.
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)

        # exit if list of names is empty (sometimes the case for benchmark)
        if len(names) == 0:
            return

        # use deltas to find which assets require sells, buys, or nothing
        to_sell = np.where(deltas < 0)[0]
        to_buy = np.where(deltas > 0)[0] # no action needed when deltas == 0

        # gather assets' current prices in an array that matches with deltas
        prices = np.array([self.assets[nm]['df']['adjOpen'][ind_all]
                           for nm in names])

        # first, sell symbols that are currently overweighted in portfolio
        for i, nm in enumerate(names[to_sell]):
            share_change = deltas[to_sell[i]] # this is negative, so...
            if main_portfolio:
                self.cash -= prices[to_sell[i]] * share_change # ...increases $$
                # only print transaction info for main portfolio
                my_pr(f"sold {abs(share_change):.0f} shares of {nm} @$"
                      f"{prices[to_sell[i]]:.2f} | ${self.cash:.2f} in account")
            else:
                self.bench_cash -= prices[to_sell[i]] * share_change # ...$$ ^
            self.assets[nm]['shares'] += share_change # ...decreases shares

        # then, buy underweighted symbols
        for i, nm in enumerate(names[to_buy]):
            share_change = deltas[to_buy[i]]
            if main_portfolio:
                self.cash -= prices[to_buy[i]] * share_change
                # only print transaction info for main portfolio
                my_pr(f"bought {share_change:.0f} shares of {nm} @$"
                      f"{prices[to_buy[i]]:.2f} | ${self.cash:.2f} in account")
            else:
                self.bench_cash -= prices[to_buy[i]] * share_change
            self.assets[nm]['shares'] += share_change

    def rebalance_portfolio(self, ind_all, ind_active, curr_rb_ind,
                            verbose=False):
        '''
        Called in `self.begin_time_loop()`.

        General method that performs a whole-portfolio rebalance by
        re-weighting core assets in-method and gets needed changes for
        satellite assets from child's rebalance_satellite(). Then, completes the
        transactions needed to restore balance.

        The hope is that this method can work with any strategy by outsourcing
        the procedures that differ in the individual rebalance_satellite()
        methods from various Strategy classes. This assumes that the
        target weights for the core will not change over time and that total
        rebalances should always try to bring the portfolio back to them.

        If that changes, perhaps add a specialized rebalance_core() method?
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)

        my_pr(f"it's a total; sat_only is {self.sat_only[curr_rb_ind]}; "
              f"${self.cash:.2f} in account")

        # get share changes for core assets
        deltas = self._get_static_rb_changes(self.core_names, ind_all)

        # get share changes for satellite assets from child's method
        deltas.extend(self.rebalance_satellite(ind_all, ind_active,
                                               curr_rb_ind, verbose=verbose))
        deltas = np.array(deltas)
        my_pr('deltas:', deltas)

        # rebalance the main (core/satellite) strategy's portfolio
        main_names = np.array(self.core_names + self.sat_names)
        self._make_rb_trades(main_names, deltas, ind_all, verbose=verbose)

        # next, get share changes for benchmark assets (CHECK FOR EXISTENCE?)
        bench_deltas = self._get_static_rb_changes(self.bench_names, ind_all,
                                                   main_portfolio=False)
        bench_deltas = np.array(bench_deltas)

        # rebalance the benchmark portfolio
        bench_names = np.array(self.bench_names)
        self._make_rb_trades(bench_names, bench_deltas, ind_all,
                             main_portfolio=False) # no printed output for now

    def _check_dividends(self, ind_all, main_portfolio=True, verbose=False):
        '''
        Called in `self.begin_time_loop()`.

        Checks whether assets currently held in a portfolio are paying out
        dividends on a given day. If so, accepts the dividend as partial shares
        of that asset if `self.reinvest_dividends` is True, or as cash if False.

        Note that this check happens before any rebalancing transactions because
        one needs to have owned an asset on the day before the dividend
        (the "ex-date") in order to claim the payment.

        `ind_all` is the integer index of each ticker's historical DataFrame
        from which to get dividend data. It aligns with self.all_dates.

        If `main_portfolio` == True (default), the method checks for dividends
        from the assets in the  main strategy's core/satellite portfolio. If
        `main_portfolio` == False, it checks assets in the benchmark portfolio.

        `verbose` is a boolean that controls whether or not to print
        information about dividends received.

        TECHNICALLY, THE ASSET NEEDS TO HAVE BEEN BOUGHT 2 BUSINESS DAYS AGO, NOT 1...
        MAKE THAT CHANGE? NOT SURE IF I WANT TO SAVE SHARE INFO OVER TIME
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)

        # choose assets to check for dividends
        tickers = (self.core_names + self.sat_names if main_portfolio
                   else self.bench_names)

        # check each asset for a dividend payment on the indicated day
        for tk in tickers:
            # if there's none, skip to the next ticker
            div_cash = self.assets[tk]['df'].loc[ind_all, 'divCash']
            if div_cash == 0:
                continue

            # if this ticker isn't currently in the portfolio, skip to the next
            shares_held = self.assets[tk]['shares']
            if shares_held == 0:
                continue

            # barring those, receive the dividend as partial shares or cash
            if self.reinvest_dividends:
                tk_price = self.assets[tk]['df'].loc[ind_all, 'adjOpen']
                partials = shares_held * (div_cash / tk_price)
                my_pr(f"**** on {self.today.strftime('%Y-%m-%d')}\n"
                      f"received a ${div_cash:.2f} dividend from {tk} @$"
                      f"{tk_price:.2f} | {partials:.4f} new shares in account")
                self.assets[tk]['shares'] += partials
            else:
                add_cash = shares_held * div_cash
                if main_portfolio:
                    self.cash += add_cash
                    my_pr(f"**** on {self.today.strftime('%Y-%m-%d')}\n"
                          f"received a ${div_cash:.2f} dividend from {tk}'s "
                          f"{shares_held} shares | ${self.cash:.2f} in account")
                else:
                    self.bench_cash += add_cash

    def begin_time_loop(self, verbose=False):
        '''
        Called in __init__ of HistoricalSimulator or by user????

        Step through all avilable dates in the historical data set, tracking and
        rebalancing the portfolio along the way. Buy at open, track stats at
        close.

        `verbose` is a boolean that controls whether or not to print the
        simulation's progress over time.

        MIGHT BE NICE TO BE ABLE TO PROVIDE A START DATE AND END DATE, CALCULATE REBALANCES BASED ON THOSE, THEN RUN THE SIMULATION OVER THAT SUBSET.
        WOULD HAVE TO RE-RUN _calc_rebalance_info() IN HERE AND REMAKE rb_indices AND sat_only.
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)

        # make lists to track values over time
        to_strategy_results = []
        to_bench_results = []
        to_cash_over_time = []

        # run the simulation
        curr_rb_ind = 0
        go = time.time()
        for i, today in enumerate(self.all_dates):
            self.today = pd.to_datetime(today)
            # "PRE-OPEN": update daily indicators based on YESTERDAY's CLOSES
            if i >= self.burn_in:
                self.on_new_day(i, i - self.burn_in)

            # "PRE-OPEN": cash in dividends from ex-date (YESTERDAY's) holdings,
            # once for main portfolio and once for benchmark
            if i >= self.burn_in:
                self._check_dividends(i, verbose=verbose)
                self._check_dividends(i, main_portfolio=False, verbose=False)

            # AT OPEN: rebalance if needed
            if i == self.rb_indices[curr_rb_ind]:
                # make rebalance calculations based on YESTERDAY'S STATS,
                # which come at index [i - burn_in] of smas, stds, etc.

                try:
                    my_pr('**** on', self.today.strftime('%Y-%m-%d'),
                          '\nvol streak: ', self.vol_streak,
                          'can enter:', self.can_enter,
                          'days out:', self.days_out)
                except AttributeError:
                    my_pr('**** on', self.today.strftime('%Y-%m-%d'))

                if self.sat_only[curr_rb_ind] == True:
                    # rebalance satellite portion
                    self.rebalance_satellite(i, i - self.burn_in, curr_rb_ind,
                                             verbose=verbose)
                else:
                    # rebalance total portfolio
                    self.rebalance_portfolio(i, i - self.burn_in, curr_rb_ind,
                                             verbose=verbose)

                if curr_rb_ind < len(self.rb_indices) - 1:
                    curr_rb_ind += 1

            # AT CLOSE: track stats
            if i >= self.burn_in:
                # save the main core/satellite portfolio's value at day's close
                pf_value = self.portfolio_value(i, rebalance=False)
                to_strategy_results.append(pf_value)

                # save the benchmark portfolio's value at day's close
                bench_pf_value = self.portfolio_value(i, main_portfolio=False,
                                                      rebalance=False)
                to_bench_results.append(bench_pf_value)

                # save amount of free cash left in the main portfolio
                to_cash_over_time.append(self.cash)

                # tried assigning to DataFrames with .loc on advice from
                # https://stackoverflow.com/a/45983830
                # turns out it's the fastest pandas assignment method, but list
                # append and numpy array assignment are 200x faster in this case

        my_pr(f"{time.time() - go:.3f} s for time loop")

        # fill in the DataFrames tracking different values over time
        self.strategy_results['value'] = to_strategy_results
        self.bench_results['value'] = to_bench_results
        self.cash_over_time['value'] = to_cash_over_time

    def plot_results(self, show_benchmark=True, logy=False,
                     return_plot=False, verbose=True):
        '''
        View a plot of your strategy's performance after the simulation is done.

        `show_benchmark` is a boolean that controls whether (True, default) or
        not (False) to display the benchmark portfolio's performance for
        purposes of comparison.

        `logy` is a boolean that controls whether the y-axis (account value in
        dollars) is on a logarithmic (True) or linear (False, default) scale.

        `return_plot` is a boolean that returns the matplotlib figure the plot
        is drawn on if there are other modifications you'd like to make to it.
        It is False by default.

        `verbose` is a boolean that controls whether final portfolio results and
        holdings are printed out.
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)

        # plot the main strategy's results
        core_pct = self.core_frac * 1e2
        sat_pct = self.sat_frac * 1e2
        lw = 2.5 if len(self.active_dates) < 1000 else 2
        ax = self.strategy_results.plot('date', 'value',
                                        label=(f"{core_pct:.0f}% core, "
                                               f"{sat_pct:.0f}% sat"),
                                        figsize=(10,10), fontsize=14,
                                        c='#beaa7c', lw=lw, logy=logy)
        # https://www.color-hex.com/color/d4bd8a, darker shade

        # print info on main final portfolio holdings and overall values
        final = len(self.all_dates) - 1 # the last index of the value array
        my_pr('end date main portfolio value: '
              f"${self.portfolio_value(final):,.02f}")

        my_pr('end date main portfolio holdings: ')
        strategy_assets = ([f"{key}: {info['shares']:.1f}"
                            for key, info in self.assets.items()
                            if info['label'] != 'benchmark'])
        my_pr(', '.join(strategy_assets), end='\n\n')

        # check if a benchmark exists; skip the benchmark plot if it doesn't
        if len(self.bench_names) == 0:
            show_benchmark = False

        # plot the benchmark strategy's results and print info, if requested
        if show_benchmark:
            self.bench_results.plot(x='date', y='value', label='benchmark',
                                    lw=lw, c='#82b6a5', ax=ax)
            # https://www.color-hex.com/color/b7e4cf, darker shade

            my_pr('end date benchmark portfolio value: '
                  f"${self.portfolio_value(final, main_portfolio=False):,.02f}")

            my_pr('end date benchmark portfolio holdings: ')
            bench_assets = ([f"{key}: {info['shares']:.1f}"
                            for key, info in self.assets.items()
                            if info['label'] == 'benchmark'])
            my_pr(', '.join(bench_assets))

        # plot a line showing the starting amount of cash
        ax.axhline(self._starting_cash, linestyle='--', c='k', alpha=.5)

        # modify some plot settings
        ax.legend(fontsize=14)
        ax.grid()

        # add dollar signs and comma separators to y ticks
        # (adapted from https://stackoverflow.com/a/25973637 and
        #  https://stackoverflow.com/a/10742904)
        ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.get_yaxis().set_minor_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # return the figure, if requested
        if return_plot:
            plt.close()
            return ax

        plt.show()

    def plot_assets(self, tickers, start_value=None, reinvest_dividends=False,
                    logy=False, return_plot=False, verbose=True):
        '''
        View a plot of assets' individual performances over the course of the
        simulation.

        `tickers` is a list with at least one of the tickers included in
        `self.assets`.

        `start_value` is a float representing the amount of money invested in
        each asset on day 1. Its default value is the original value chosen
        for `self.cash` when this class instance was initialized.

        `reinvest_dividends` is a boolean that, if True, reinvests any dividend
        income back into the asset that paid it out. Coming soon?

        `logy` is a boolean that controls whether the y-axis (account value in
        dollars) is on a logarithmic (True) or linear (False, default) scale.

        `return_plot` is a boolean that returns the matplotlib figure the plot
        is drawn on if there are other modifications you'd like to make to it.
        It is False by default.

        `verbose` is a boolean that controls whether final portfolio results and
        holdings are printed out.
        '''
        # confirm that arguments are acceptable
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)
        if not isinstance(tickers, list):
            raise ValueError('make sure that `tickers` is a list.')
        for tk in tickers:
            if tk not in self.assets.keys():
                raise ValueError(f"{tk} is not part of your list of assets.")
        if start_value is None:
            start_value = self._starting_cash
        if reinvest_dividends:
            raise NotImplementedError('Coming soon...')

        # make separate colormaps for each ticker label
        if len(self.core_names) != 0:
            core_cmap = plt.cm.viridis.colors[250:150:-1]
            core_cmap = core_cmap[::len(core_cmap) // len(self.core_names)]
        if len(self.sat_names) != 0:
            sat_cmap = plt.cm.plasma.colors[225:125:-1]
            sat_cmap = sat_cmap[::len(sat_cmap) // len(self.sat_names)]
        if len(self.bench_names) != 0:
            bench_cmap = plt.cm.twilight.colors[200:100:-1]
            bench_cmap = bench_cmap[::len(bench_cmap) // len(self.bench_names)]

        # make the plot, asset by asset
        fig, ax = plt.subplots(figsize=(10, 10))
        lw = 1.5#2.5 if len(self.active_dates) < 1000 else 2
        for i, tk in enumerate(tickers):
            # adjust ticker's historical prices so first day is at start_value
            tick_info = self.assets[tk]['df'].loc[:, ['date','adjClose']].copy()
            tick_info.loc[:, 'adjClose'] /= tick_info.loc[0, 'adjClose']
            tick_info.loc[:, 'adjClose'] *= start_value

            # find out what type of asset this ticker is and assign color
            tick_type = self.assets[tk]['label']
            if tick_type == 'core':
                col = core_cmap.pop()
            elif tick_type == 'satellite':
                col = sat_cmap.pop()
            elif tick_type == 'benchmark':
                col = bench_cmap.pop()
            else:
                raise ValueError(f"invalid 'label' for ticker {tk}")

            # add this ticker's data to the plot
            tick_info.plot('date', 'adjClose',
                                label=f"{tk} ({tick_type})",
                                lw=lw, c=col,
                                fontsize=14, logy=logy, ax=ax)

            # print final value of ticker holdings
            my_pr(f"{tk} ending value: "
                  f"${tick_info.iloc[-1]['adjClose']:,.2f}")

        # plot a line showing the starting investment value
        ax.axhline(start_value, linestyle='--', c='k', alpha=.5)

        # modify some plot settings
        ax.legend(fontsize=14)
        ax.grid()

        # add dollar signs and comma separators to y ticks
        # (adapted from https://stackoverflow.com/a/25973637 and
        #  https://stackoverflow.com/a/10742904)
        ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.get_yaxis().set_minor_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # return the figure, if requested
        if return_plot:
            plt.close()
            return ax

        plt.show()
