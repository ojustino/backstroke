#!/usr/bin/python3
from better_abc import ABC, abstract_attribute, abstractmethod
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

class HistoricalSimulator(ABC):
    '''
    The parent of a Strategy class that does the heavy lifting in simulating
    how a portfolio composed of a PortfolioMaker instance's assets would have
    performed over a specified period of time. Run self.begin_time_loop() after
    initializing an instance to run a simulation; each instance is good for one
    simulation only.

    Handles downloading asset data from Tiingo, rebalancing (whole portfolio
    and satellite-only through its child), dividends, and plotting of
    post-simulation results. The simulation includes both the main core/
    satellite portfolio and a standard benchmark portfolio.

    Arguments
    ---------

    Portfolio : `portfolio_maker.PortfolioMaker`, required
        A PortfolioMaker instance whose `assets` attribute contains your desired
        assets, fractions, and categories. Its `check_assets()` method must pass
        before you can run any simulations.

    cash : float, optional
        The amount of unspent money in your portfolio at the start of the
        simulation. Note that this is separate from the value of any initial
        shares held in Portfolio.assets. [default: $10,000]

    start_date : `datetime.datetime`, optional
        The first trading date in your simulation. If the market wasn't open on
        your chosen date, a nearby date will be chosen.
        [default: datetime.datetime(2007, 5, 22)]

    end_date : `datetime.datetime`, optional
        The last trading date in your simulation. If the market wasn't open on
        your chosen date, a nearby date will be chosen.
        [default: datetime.datetime(2015, 5, 22)]

    sat_rb_freq : float, optional
        The number of times per year to rebalance the satellite portion of your
        portfolio. Allowed rebalance frequencies are 1, 2, 3, 4, 6, 8, and 12
        times per year, as well as 365.25 (daily). [default: 6]

    tot_rb_freq : float, optional
        The number of times per year to rebalance the entire portfolio, core and
        satellite. This value must be less than or equal to `sat_rb_freq`.
        Allowed rebalance frequencies are 1, 2, 3, 4, 6, 8, and 12 times per
        year. [default: 1]

    target_rb_day : integer, optional
        For rebalance frequencies of one month or more, the market day of the
        month on which you'd like rebalances to take place. Uses list-style
        indexing, so both positive and negative values are acceptable as long as
        their absolute value is 10 or lower. [default: -2]

    reinvest_dividends : boolean, optional
        When True, any dividends paid out by an asset are used immediately to
        purchase partial shares of that asset. When False, dividends are taken
        in as cash and spent on the next rebalance date. [default: False]
    '''
    # earliest start dates: 1998-11-22, 2007-05-22, 2012-10-21
    def __init__(self, Portfolio, cash=1e4,
                 start_date=datetime(2007, 5, 22),
                 end_date=datetime(2007, 5, 22) + timedelta(days=365.25*8),
                 sat_rb_freq=6, tot_rb_freq=1,
                 target_rb_day=-2, reinvest_dividends=False):
        # make sure a PortfolioMaker object is present
        if not isinstance(Portfolio, PortfolioMaker):
            raise ValueError('The first argument of HistoricalSimulator() must '
                             'be a PortfolioMaker() instance.')

        # ensure that target_rb_day is valid; save it if so
        if abs(target_rb_day) > 10:
            raise ValueError('The absolute value of `target_rb_date` must be '
                             'less than or equal to 10.')
        elif type(target_rb_day) != int:
            raise ValueError('`target_rb_date` must be an integer.')
        else:
            self._target_rb_day = target_rb_day

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

        # track the current simulation date
        self.today = self.open_date

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
        self._bench_cash = self.portfolio_value(rebalance=True)
        self._starting_value = self._bench_cash

        # save the core and satellite fractions
        self.sat_frac = np.round(Portfolio.sat_frac, 6)
        self.core_frac = np.round(1 - self.sat_frac, 6)

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

    # define attribute and methods that must be present a child Strategy class
    # **(make sure to use the listed arguments)**
    @abstract_attribute
    def burn_in(self):
        '''
        An attribute representing the number of days of data needed before a
        Strategy class can begin trading. For example, a Strategy based on a
        200-day simple moving average of some asset's price needs `burn_in=200`.
        '''
        pass

    @abstractmethod
    def on_new_day(self, ind_all, ind_active):
        '''
        Called in HistoricalSimulator.begin_time_loop().

        Keeps daily track of whatever indicators are needed to carry out a
        Strategy. See SMAStrategy() for an example, though this method can also
        just be a simple `pass` statement (as in VolTargetStrategy()) if
        there's nothing that must be tracked daily.

        Arguments
        ---------

        ind_all : integer, required
            The index of each ticker's historical DataFrame from which to get
            the price data. It aligns with self.all_dates.

        ind_active : integer, required
            The index of self.active_dates that matches the simulation's current
            date. ind_all = ind_active + self.burn_in.
        '''
        pass

    @abstractmethod
    def rebalance_satellite(self, ind_all, ind_active, curr_rb_ind,
                            verbose=False):
        '''
        Called in HistoricalSimulator.rebalance_portfolio() or
        HistoricalSimulator.begin_time_loop().

        A satellite-only version of self._get_static_rb_changes() that
        re-weights the main portfolio's satellite assets according to an
        individual Strategy's logic. Either returns a list of share changes on
        total rebalaces or turns that list into an array that's passed to
        self._make_rb_trades() on satellite-only rebalances. In either
        case, the in-market asset's change in shares should be the first entry.

        Arguments
        ---------

        ind_all : integer, required
            The index of each ticker's historical DataFrame from which to get
            the price data. It aligns with self.all_dates.

        ind_active : integer, required
            The index of self.active_dates that matches the simulation's current
            date. ind_all = ind_active + self.burn_in.

        curr_rb_ind : integer, required
            The index of self.rb_indices and self.sat_only.
            self.sat_only[curr_rb_ind] is True on satellite-only rebalances and
            False on total rebalances. self.rb_indices[curr_rb_ind] == ind_all.

        verbose : boolean, optional
            Controls whether or not to print any debugging information you
            choose to include in this method. [default: False]
        '''
        pass

    # define HistoricalSimulator's own methods
    def portfolio_value(self, ind_all=None,
                        main_portfolio=True, rebalance=False):
        '''
        Return the value of all assets currently held in the portfolio,
        including cash.

        Arguments
        ---------

        ind_all : integer, required
            The index of each ticker's historical DataFrame from which to get
            the price data. It aligns with self.all_dates.

        main_portfolio : boolean, optional
            If True, the method returns the value of main strategy's core/
            satellite portfolio. If False, the method returns the value of the
            benchmark portfolio. [default: True]

        rebalance : boolean, optional
            If False, asset prices use the current day's (self.today) closing
            price. If True, assets are valuated using the current day's opening
            price. [default: False]
        '''
        if not isinstance(rebalance, bool):
            raise ValueError("'rebalance' must be a bool.")

        # get remaining cash for the chosen portfolio
        cash = self.cash if main_portfolio else self.bench_cash

        # if sim hasn't reached the first trading date, return portfolio value
        # on the first trading date
        if ind_all is None and self.today < self.start_date:
            ind_all = self.burn_in
            # rebalance should be set to True in this case...

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
        Called in self._build_assets_dict(), but can also be used independently.

        Download an asset's historical price data from Tiingo, convert the
        result to a pandas DataFrame, then return it.

        Arguments
        ---------

        tick : str, required
            The ticker of the asset whose data will be downloaded.

        open_date : `datetime.datetime`, required
            The earliest date of historical data to be downloaded. (Note that
            when this method is called upon initializing HistoricalSimulator,
            this argument is self.open_date, not self.start_date.)

        end_date : `datetime.datetime`, optional
            The final date of historical data to be downloaded. (When this
            method is called upon initializing HistoricalSimulator, this
            argument is self.end_date.) [default: datetime.datetime.now()]
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

        Arguments
        ---------

        tick_info : `pandas.core.frame.DataFrame`, required
            A DataFrame with start/end date and asset type (stock, ETF, mutual
            fund) information. Comes from the `tick_info` attribute of the
            PortfolioMaker object used to create the current HistoricalSimulator
            instance.
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

        Verifies that proposed assets exist over the range of self.open_date to
        self.end_date. If so, retrieves their historical price data from Tiingo.

        Then, adds 'df' and 'shares' keys to each assets[TICKER] dict; their
        respective values are the returned DataFrame and the number of 'ticker'
        shares currently held (0 to start).

        Arguments
        ---------

        Portfolio : `portfolio_maker.PortfolioMaker`, required
            The PortfolioMaker object provided when initializing this
            HistoricalSimulator instance.
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

        # ensure that each asset has the same number of dates
        num_dates = np.unique([len(assets[nm]['df']['date']) for nm in assets])
        assert len(num_dates) == 1, 'some ticker DataFrames are missing dates'

        return assets

    def _get_date_arrays(self):
        '''
        Called in __init__() of HistoricalSimulator.

        Traverses downloaded historical data and returns an array with all
        available dates (self.all_dates) and another with the burn-in dates
        removed (self.active_dates).

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

    def _modify_rb_vars(self, rb_inds, sat_only, ind, is_sat_only_rb=False):
        '''
        Called in self._calc_rebalance_info().

        `rb_inds` and `sat_only` can start out as lists or arrays. This method
        handles that ambiguity by first trying to append `ind` to `obj`. (i.e.,
        the case where self.rb_indices and self.sat_only grow one item at a time
        because self.sat_rb_freq == 365.25.)

        If append() is not an attribute of `rb_inds` or `sat_only`, the method
        pivots to flipping the value of index `ind` of `sat_only`, since it and
        `rb_inds` must be arrays instead. (This is the case where both are
        pre-filled and the latter has indices to change from True to False.)

        See self._calc_rebalance_info() for more information on how
        self.rb_indices and self.sat_only are built.

        Arguments
        ---------

        rb_inds, sat_only : list or `numpy.ndarray`, required
            Type depends on this class instance's satellite rebalance frequency.

        ind : integer, required
            The index of `rb_indices`/`sat_only` to be appended or flipped.

        is_sat_only_rb: boolean, optional
            The type of rebalance that will happen on the date that corresponds
            with `ind` in each asset's historical DataFrame. If True, it's a
            satellite-only rebalance; if False, it's for the total portfolio.
            [default: False]
        '''
        # need to check if ind is None (final month scenario)
        if ind is not None:
            try:
                rb_inds.append(ind)
                sat_only.append(is_sat_only_rb)
            except AttributeError:
                sat_only[ind] = is_sat_only_rb
                # (no changes needed in array version of rb_inds)

        return rb_inds, sat_only

    def _last_day_of_month(self, year, month):
        '''
        Called in self._calc_rebalance_info().

        Reliably calculate the date of the specified month's final market day.

        Arguments
        ---------

        year : integer, required
            The year for the date in question.

        month : integer, required
            The month for the date in question.
        '''
        next_month = datetime(year, month, 28) + timedelta(days=4)
        return next_month - timedelta(days=next_month.day)

    def _get_mth_rb_range(self, yr, mth):
        '''
        Called in self._calc_rebalance_info().

        Returns the first and last days of a range of valid, weekday-only
        rebalance dates for a particular year/month. Based on user's preference
        for days after the first of a month (or days before the last market day
        of the month) to rebalance.

        The size of the range depends on the buffer used; if the buffer is 0,
        the first and last days will be the same. However, use of a buffer is
        advised since the desired date might fall on a holiday in a given month.

        Arguments
        ---------

        yr : integer, required
            The year to consider in calculating rebalance dates.

        mth: integer, required (from 1 to 12)
            The month to consider in calculating rebalance dates.
        '''
        # SHOULD buffer BE AN ARGUMENT?

        # assign target rb date index, reference date, and iteration direction
        # based on whether rb date is counted from the month's beginning or end
        if self._target_rb_day < 0:
            rb_day = -self._target_rb_day - 1
            ref_day = self._last_day_of_month(yr, mth)
            iter_day = -1
        else:
            rb_day = self._target_rb_day
            ref_day = datetime(yr, mth, 1)
            iter_day = 1

        # create object to hold beginning/end dates of range
        date_range = [] # for short lists, min(list) is faster than array.min()

        # set number for market days to capture in range beyond exact rb date
        buffer = 2

        # set initial loop date and counter days before/after ref_day
        dt = ref_day
        days_beyond_ref = 0

        while True:
            # only count weekdays as possible options
            if dt.isoweekday() < 6:
                if days_beyond_ref == rb_day:
                    date_range.append(np.datetime64(dt))
                elif days_beyond_ref == rb_day + buffer:
                    date_range.append(np.datetime64(dt))
                    break
                # iterate the loop's "days beyond rb_day" counter
                days_beyond_ref += 1

            # iterate the loop's date
            dt += timedelta(days=iter_day)

        return date_range

    def _calc_rebalance_info(self):
        '''
        Called in __init__() of HistoricalSimulator.

        Uses satellite and total portfolio rebalance frequencies to get
        self.rb_indices, an array of the indices of each asset's DataFrame that
        will trigger rebalance events.

        Returns that along with self.sat_only, an associated, same-size array
        that is True when the rebalance is for the satellite portion only and
        False when it's time for a full portfolio rebalance.

        Non-daily frequencies will rebalance on the penultimate market day of a
        qualifying month to try and avoid whipsawing from larger investors doing
        their own rebalancing on the last or first market day of the month.
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
            rb_inds = []
            sat_only = []

        else: # if daily... (only. in future could do every 2, 3 days and so on)
            # ...then every month has rebalance events
            sat_mths = all_months
            print('sat', sat_mths, '\ntot', tot_mths)

            # include every active_date as a possible rebalance date
            # (total rebalance days will be flipped to True in sat_only later)
            rb_inds = np.arange(len(self.active_dates))
            sat_only = np.ones(len(self.active_dates)).astype(bool)

        #go = time.time()
        yr = self.start_date.year
        while yr <= self.end_date.year:
            # make array with all eligible months
            months = np.arange(1 if yr != self.start_date.year
                               else self.start_date.month,
                               13 if yr != self.end_date.year
                               else self.end_date.month + 1)
            # end month may not reach a reblance date, but allow for it if so

            print(months, yr)
            # limit months to those cleared for rebalance events
            eligible = [mth for mth in months
                        if mth in tot_mths or mth in sat_mths]

            # check every month in the current year...
            for mth in eligible:
                # automatically make start_date a total rebalance event
                if (yr == self.start_date.year
                    and mth == self.start_date.month):
                    (rb_inds,
                     sat_only) = self._modify_rb_vars(rb_inds, sat_only, 0,
                                                      is_sat_only_rb=False)
                # in subsequent months, find desired market day for rebalancing
                else:
                    # get first and last possible rebalance days (using a range
                    # instead of a specific day for protection against holidays)
                    fnl = self._get_mth_rb_range(yr, mth)

                    # save dates that fall within that range
                    poss = np.where((min(fnl) <= self.active_dates)
                                    & (self.active_dates <= max(fnl)))[0]

                    # save the last day in the range as this month's rb date
                    try:
                        day = poss[-1]
                    # if there are no dates in that range, use None instead
                    # (i.e., active_dates' last month cuts off prior to rb date)
                    except IndexError:
                        day = None
                        # NOTE: depending on buffer size in _get_mth_rb_range(),
                        # rb's could be triggered if sim ends within buffer but
                        # before target date. not yet sure how to fix this...

                    # update class' rb objects with this month's info
                    kind = True if mth not in tot_mths else False
                    (rb_inds,
                     sat_only) = self._modify_rb_vars(rb_inds, sat_only, day,
                                                      is_sat_only_rb=kind)

            yr += 1
        #print(f"{time.time() - go:.3f} s for rebalance info loop")

        # make arrays and shift rb_indices to account for burn-in days
        rb_indices = np.array(rb_inds) + self.burn_in
        sat_only = np.array(sat_only)

        return rb_indices, sat_only

    def _get_static_rb_changes(self, names, ind_all, main_portfolio=True):
        '''
        Called in self.rebalance_portfolio().

        Returns an array with the changes in shares for all non-satellite assets
        in the portfolio in question. These (core or benchmark) assets are
        "static" because their target allocations do not change over time.

        Arguments
        ---------

        names : list, required
            A list of assets who share the same label; it should typically
            either be self.core_names or self.bench_names.

        ind_all : integer, required
            The integer index of each ticker's historical DataFrame from which
            to get the price data. It aligns with self.all_dates.

        main_portfolio : boolean, optional
            If True, the method finds share changes for the main strategy's
            core/satellite portfolio. If False, the method finds share changes
            for the benchmark portfolio. [default: True]
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
        Called in self.rebalance_portfolio() or the child Strategy class'
        rebalance_satellite().

        Completes the transactions needed to rebalance a portfolio.

        Arguments
        ---------

        names : `numpy.ndarray`, required
            The string tickers of the assets that will be rebalanced.
            If called from rebalance_portfolio(), it should be:
                -- np.array(self.core_names + self.sat_names)
            If called from rebalance_satellite():
                -- np.array(self.sat_names)
            If called from rebalance_portfolio() and `main_portfolio` is False:
                -- np.array(self.bench_names)

        deltas : `numpy.ndarray`, required
            The corresponding share changes for the assets in `names`.

        ind_all : integer, required
            The index of each ticker's historical DataFrame from which to get
            the price data. It aligns with self.all_dates.

        main_portfolio : boolean, optional
            If True, rebalances the main strategy's core/satellite portfolio.
            If False, rebalances the benchmark portfolio. [default: True]

        verbose : boolean, optional
            If True, the method prints information on completed trades.
            [default: False]
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
        Called in self.begin_time_loop().

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

        Arguments
        ---------

        ind_all : integer, required
            The index of each ticker's historical DataFrame from which to get
            the price data. It aligns with self.all_dates.

        ind_active : integer, required
            The index of self.active_dates that matches the simulation's current
            date. ind_all = ind_active + self.burn_in.

        curr_rb_ind : integer, required
            The index of self.rb_indices and self.sat_only.
            self.sat_only[curr_rb_ind] is True on satellite-only rebalances and
            False on total rebalances. self.rb_indices[curr_rb_ind] == ind_all.

        verbose : boolean, optional
            If True, the method prints information on completed trades.
            [default: False]
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

        # next, get share changes for benchmark assets
        bench_deltas = self._get_static_rb_changes(self.bench_names, ind_all,
                                                   main_portfolio=False)
        bench_deltas = np.array(bench_deltas)

        # rebalance the benchmark portfolio
        bench_names = np.array(self.bench_names)
        self._make_rb_trades(bench_names, bench_deltas, ind_all,
                             main_portfolio=False) # no printed output for now

    def _check_dividends(self, ind_all, main_portfolio=True, verbose=False):
        '''
        Called in self.begin_time_loop().

        Checks whether assets currently held in a portfolio are paying out
        dividends on a given day. If so, accepts the dividend as partial shares
        of that asset if self.reinvest_dividends is True, or as cash if False.

        Note that this check happens before any rebalancing transactions because
        one needs to have owned an asset on the day before the dividend
        (the "ex-date") in order to claim the payment.

        (Technically, you need to have owned the asset two business days before
        the dividend payment date, but this only matters for assets that are
        rebalanced daily. I likely won't go to this level of specificity.)

        Arguments
        ---------

        ind_all : integer, required
            The index of each ticker's historical DataFrame from which to get
            the price data. It aligns with self.all_dates.

        main_portfolio : boolean, optional
            If True, rebalances the main strategy's core/satellite portfolio.
            If False, rebalances the benchmark portfolio. [default: True]

        verbose : boolean, optional
            If True, prints information when dividends are received in the
            simulation. [default: False]
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
        rebalancing the portfolio along the way. Buys stocks based on opening
        prices and tracks stats based on closing prices.

        MIGHT BE NICE TO BE ABLE TO PROVIDE A START DATE AND END DATE, CALCULATE REBALANCES BASED ON THOSE, THEN RUN THE SIMULATION OVER THAT SUBSET.
        WOULD HAVE TO RE-RUN _calc_rebalance_info() IN HERE AND REMAKE self.rb_indices AND self.sat_only.

        Arguments
        ---------

        verbose : boolean, optional
            If True, prints the simulation's progress over time.
            [default: False]
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

    def _set_log_axis(self, ax, results):
        '''
        Called in self.plot_results() and self.plot_assets().

        Ensures that minimum and maximum y values on the y axis of a logarithmic
        plot are clearly labeled. The methods above place labels at 1, 2.5, 5,
        and 7.5 of every order of magnitude, so this one finds the next
        multiple up and down and makes sure they appear on the plot.

        Arguments
        ---------

        ax : `matplotlib.axes._subplots.AxesSubplot`, required
            The axis object that contains the plot being generated.

        results : `numpy.ndarray`, required
            Contains all values that will be plotted on `ax`. The structure
            doesn't matter; only the minimum and maximum values are used.
        '''
        # set minor label locator for y (dollar value) axis
        ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10,
                                                         subs=(2.5, 5, 7.5)))

        # find the max/min values in results, separate them into their bases and
        # exponents in scientific notation, then round up/down to next labels
        max_result = results.max()
        max_oom = np.floor((np.log10(max_result)))
        max_base = max_result / 10**max_oom
        rounded_max_base = np.ceil(max_base / 2.5) * 2.5

        min_result = results.min()
        min_oom = np.floor((np.log10(min_result)))
        min_base = min_result / 10**min_oom
        rounded_min_base = np.floor(min_base / 2.5) * 2.5
        if rounded_min_base == 0: # avoid taking log(0)
            rounded_min_base = 1

        # make a change if the current y maximum is too low
        target_y_max = rounded_max_base * 10**max_oom
        curr_y_max = ax.axis()[-1]
        if curr_y_max < target_y_max:
            ax.set_ylim(ax.get_ylim()[0], target_y_max)

        # make a change if the current y minimum is too high
        target_y_min = rounded_min_base * 10**min_oom
        curr_y_min = ax.axis()[-2]
        if curr_y_min > target_y_min:
            ax.set_ylim(target_y_min, ax.get_ylim()[-1])

        return ax

    def plot_results(self, show_benchmark=True, logy=False,
                     return_plot=False, verbose=True):
        '''
        View a plot of your strategy's performance after the simulation is done.

        Arguments
        ---------

        show_benchmark : boolean, optional
            If True, displays the benchmark portfolio's performance for purposes
            of comparison. [default: True]

        logy : boolean, optional
            If True, the y-axis (account value in dollars) will have a
            logarithmic scale. If False, the scale will be linear.
            [default: False]

        return_plot : boolean, optional
            If True, returns the matplotlib axes object the plot is drawn on in
            case you'd like to modify it further. [default: False]

        verbose : boolean, optional
            If True, the method prints final portfolio values and holdings.
            [default: True]
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

        my_pr('end date main portfolio shares: ')
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

            my_pr('end date benchmark portfolio shares: ')
            bench_assets = ([f"{key}: {info['shares']:.1f}"
                            for key, info in self.assets.items()
                            if info['label'] == 'benchmark'])
            my_pr(', '.join(bench_assets))

        # plot a line showing the starting amount of cash
        ax.axhline(self._starting_value, linestyle='--', c='k', alpha=.5)

        # if logarithmic, ensure the axes are properly populated
        if logy:
            results = (np.stack((self.strategy_results.loc[:, 'value'],
                                 self.bench_results.loc[:, 'value']))
                       if show_benchmark
                       else self.strategy_results.loc[:, 'value'])
            ax = self._set_log_axis(ax, results)

        # modify other plot settings
        ax.legend(fontsize=14)
        ax.grid(which='both')

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

    def plot_assets(self, *tickers, start_value=None, reinvest_dividends=False,
                    logy=False, return_plot=False, verbose=True):
        '''
        View a plot of one or more assets' individual performances over the
        course of the simulation.

        Arguments
        ---------

        *tickers : str, required (at least one)
            At least one ticker name. All provided tickers must be among those
            in self.assets or an error is thrown.

        start_value : float, required
            The amount of money invested in each asset on day 1. Its default
            value is the original value chosen for self.cash when this instance
            was initialized.

        reinvest_dividends : boolean, optional
            (Coming soon?) If True, reinvests any dividend income back into the
            asset that paid it out. [default: False]

        logy : boolean, optional
            If True, the y-axis (account value in dollars) will have a
            logarithmic scale. If False, the scale will be linear.
            [default: False]

        return_plot : boolean, optional
            If True, returns the matplotlib axes object the plot is drawn on in
            case you'd like to modify it further. [default: False]

        verbose : boolean, optional
            If True, the method prints final value of each asset's holdings.
            [default: True]
        '''
        # confirm that arguments are acceptable
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)
        for tk in tickers:
            if tk not in self.assets.keys():
                raise ValueError(f"{tk} is not part of your list of assets.")
        if start_value is None:
            start_value = self._starting_value
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
        results = []
        for i, tk in enumerate(tickers):
            # adjust ticker's historical prices so first day is at start_value
            tick_info = self.assets[tk]['df'].loc[:, ['date','adjClose']].copy()
            tick_info = tick_info.loc[self.burn_in:]
            tick_info.index = pd.RangeIndex(len(tick_info.index))
            tick_info.loc[:, 'adjClose'] /= tick_info.loc[0, 'adjClose']
            tick_info.loc[:, 'adjClose'] *= start_value
            results.append(tick_info.loc[:, 'adjClose'])

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
                           lw=1.5, c=col,
                           fontsize=14, logy=logy, ax=ax)

            # print final value of ticker holdings
            my_pr(f"{tk} ending value: "
                  f"${tick_info.iloc[-1]['adjClose']:,.2f}")

        # plot a line showing the starting investment value
        ax.axhline(start_value, linestyle='--', c='k', alpha=.5)

        # if logarithmic, esnure the axes are properly populated
        if logy:
            results = np.array(results)
            ax = self._set_log_axis(ax, results)

        # modify other plot settings
        ax.legend(fontsize=14)
        ax.grid(which='both')

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
