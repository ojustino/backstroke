#!/usr/bin/python3
from better_abc import ABC, abstract_attribute, abstractmethod
from datetime import timedelta
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

    start_date : `pandas.Timestamp` or `datetime.datetime`, optional
        The first trading date in your simulation. If the market wasn't open on
        your chosen date, the next market date will be chosen.
        [default: pandas.Timestamp(2007, 5, 22)]

    end_date : `pandas.Timestamp` or `datetime.datetime`, optional
        The last trading date in your simulation. If the market wasn't open on
        your chosen date, the last market date before it will be chosen.
        [default: pandas.Timestamp(2015, 5, 22)]

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
        their absolute value is 13 or lower. [default: -2]

    reinvest_dividends : boolean, optional
        When True, any dividends paid out by an asset are used immediately to
        purchase partial shares of that asset. When False, dividends are taken
        in as cash and spent on the next rebalance date. [default: False]

    verbose : boolean, optional
        Whether or not to print the download's progress. [default: False]
    '''
    # earliest start dates: 1998-11-22, 2007-05-22, 2012-10-21
    def __init__(self, Portfolio, cash=1e4,
                 start_date=pd.Timestamp(2007, 5, 22),
                 end_date=pd.Timestamp(2015, 5, 22),
                 sat_rb_freq=6, tot_rb_freq=1, target_rb_day=-2,
                 reinvest_dividends=False, verbose=False):
        # make sure a PortfolioMaker object is present
        if not isinstance(Portfolio, PortfolioMaker):
            raise ValueError('The first argument of HistoricalSimulator() must '
                             'be a PortfolioMaker() instance.')

        # ensure that target_rb_day is valid; save it if so
        if abs(target_rb_day) > 13:
            raise ValueError('The absolute value of `target_rb_date` must be '
                             'less than or equal to 13.')
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
        buffer_days = int(self.window * mkt_to_real_days) + 5
        self.open_date = pd.Timestamp(start_date - timedelta(buffer_days))

        # save dates over which analysis will take place
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        # track the current simulation date
        self.today = self.open_date

        # validate proposed asset dictionary, then add historical data to it
        self.assets = self._validate_assets_dict(Portfolio, verbose)

        # make arrays of all dates in set and all *active* dates
        self.all_dates, self.active_dates = self._get_date_arrays()

        # on which dates do rebalances occur, and are they satellite-only?
        self.rb_info = self._calc_rebalance_info(verbose)

        # save preference for handling dividend payouts
        self.reinvest_dividends = reinvest_dividends

        # track remaining money in main and benchmark portfolios
        # (are properties, so an error is thrown if they go negative)
        # (go Decimal here?)
        self._cash = float(cash)
        self._bench_cash = self.portfolio_value(self.start_date, at_close=False)
        self._starting_value = self._bench_cash

        # save the core and satellite fractions
        self.sat_frac = np.round(Portfolio.sat_frac, 6)
        self.core_frac = np.round(1 - self.sat_frac, 6)

        # make DataFrames to track portfolios and cash over time
        self._create_tracking_arrays()

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
            raise ValueError('More cash was spent than remains '
                             'in main portfolio.')
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
    def window(self):
        '''
        An attribute representing the number of days of data needed before a
        Strategy class can begin trading. For example, a Strategy based on a
        200-day simple moving average of some asset's price needs `window=200`.
        '''
        pass

    @abstractmethod
    def on_new_day(self):
        '''
        Called in HistoricalSimulator.begin_time_loop().

        Keeps daily track of whatever indicators are needed to carry out a
        Strategy. See SMAStrategy() for an example, though this method can also
        just be a simple `pass` statement (as in VolTargetStrategy()) if
        there's nothing that must be tracked daily.

        Returns
        -------

        Nothing.
        '''
        pass

    @abstractmethod
    def refresh_parent(self):
        '''
        Called in HistoricalSimulator.append_date().

        Re-creates any needed class attributes after a new date is appended to
        asset dataFrames. Should only be called *before* any simulation is run.
        If supporting appended dates isn't a concern, this method can just be a
        simple `pass` statement.

        Returns
        -------

        Nothing.
        '''
        pass

    @abstractmethod
    def rebalance_satellite(self, day, verbose=False):
        '''
        Called in HistoricalSimulator.rebalance_portfolio() or
        HistoricalSimulator.begin_time_loop().

        A satellite-only version of self._get_static_rb_changes() that
        re-weights the main portfolio's satellite assets according to an
        individual Strategy's logic.

        Arguments
        ---------

        day : `pandas.Timestamp` or `datetime.datetime`, required
            The simulation's current date. Used to find whether this rebalance
            is satellite-only or for the total portfolio.

        verbose : boolean, optional
            Controls whether or not to print any debugging information you
            choose to include in this method. [default: False]

        Returns
        -------

        On total rebalances: A list of the satellite assets' share changes. If
        there are no satellite assets, the list should be empty.

        On satellite-only rebalances: Nothing, but the method should end with a
        call to self.make_rb_trades().

        ** In either case, the in-market asset's changes should come first in
        the list/array of share changes. **
        '''
        pass

    # define HistoricalSimulator's own methods
    def portfolio_value(self, date=None, main_portfolio=True, at_close=True):
        '''
        Return the value of all assets currently held in the portfolio on a
        certain date, including cash.

        Arguments
        ---------

        date : `pandas.Timestamp` or `datetime.datetime`, optional
            The date whose price data is used in the value calculation. Will
            cause an error if the date is not between the class' open and close
            dates or if it is not a market day. [default: self.today]

        main_portfolio : boolean, optional
            If True, the method returns the value of main strategy's core/
            satellite portfolio. If False, the method returns the value of the
            benchmark portfolio. [default: True]

        at_close : boolean, optional
            If True, asset prices use the given `date`'s closing price.
            If False, assets are valuated using the current day's opening
            price; use this option for rebalances. [default: True]
        '''
        if not isinstance(at_close, bool):
            raise ValueError("'at_close' must be a bool.")

        # ensure that the requested date is within the class instance's range
        date = self.today if date is None else date
        if not self.open_date <= date <= self.end_date:
            raise ValueError("`date` must occur between `self.open_date` and "
                             "`self.end_date`")

        # get remaining cash for the chosen portfolio
        cash = self.cash if main_portfolio else self.bench_cash

        # determine whether to use open or close prices for assets
        col = 'adjClose' if at_close else 'adjOpen'

        # collect labels for assets in the chosen portfolio
        labels = {'core', 'satellite'} if main_portfolio else {'benchmark'}

        # multiply shares held of each ticker by their current prices
        holdings = np.sum([info['shares'] * info['df'].loc[date, col]
                           for info in self.assets.values()
                           if info['label'] in labels])

        return cash + holdings

    def call_tiingo(self, tick, open_date,
                    end_date=pd.Timestamp.today(), verbose=True):
        '''
        Called in self._build_assets_dict(), but can also be used independently.

        Download an asset's historical price data from Tiingo, convert the
        result to a pandas DataFrame, then return it.

        Arguments
        ---------

        tick : str, required
            The ticker of the asset whose data will be downloaded.

        open_date : `pandas.Timestamp` or `datetime.datetime`, required
            The earliest date of historical data to be downloaded. (Note that
            when this method is called upon initializing HistoricalSimulator,
            this argument is self.open_date, not self.start_date.)

        end_date : `pandas.Timestamp` or `datetime.datetime`, optional
            The final date of historical data to be downloaded. (When this
            method is called upon initializing HistoricalSimulator, this
            argument is self.end_date.) [default: pandas.Timestamp.today()]

        verbose : boolean, optional
            Whether or not to print the download's progress. [default: True]
        '''
        open_date = open_date.strftime('%Y-%m-%d') # (e.g. '1998-07-13')
        end_date = end_date.strftime('%Y-%m-%d')
        if verbose:
            print(f"{tick} from {open_date} to {end_date}...")

        url = f"https://api.tiingo.com/tiingo/daily/{tick}/prices"
        headers = {'Content-Type': 'application/json'}
        params = {
            'startDate': open_date,
            'endDate': end_date,
            'format': 'json',
            'resampleFreq': 'daily',
            'token': MY_API_KEY,
        }

        resp = requests.get(url, params=params, headers=headers)
        assert resp.status_code == 200, f"HTTP status code {resp.status_code}"

        # convert JSON to dataFrame with index column made of Timestamp objects
        df = pd.DataFrame.from_dict(resp.json())
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').tz_localize(None) # use timezone-naive dates

        return df

    def append_date(self, date, price_dict, verbose=False):
        '''
        A (somewhat hacky) way to add a new date's worth of price data to the
        'df' attribute of each asset in `self.assets`. Must be run *before*
        `self.begin_time_loop()`.

        Useful for scenarios where you'd like to do a real life rebalance but
        lack the current day's info since Tiingo only updates its data after
        the market closes.

        Arguments
        ---------

        date : `pandas.Timestamp` or `datetime.datetime`, required
            The new date to add to the simulation. Must be a weekday and must
            not be timezone-aware.

        price_dict : dict, required
            A dictionary whose keys are the string ticker symbols of each asset
            in the current object (must match the keys in `self.assets`) and
            whose values are floats representing the assets' prices on `date`.

        verbose : boolean, optional
            Whether to print rebalance month information. [default: False]
        '''
        # ensure the proposed date is valid
        date = pd.Timestamp(date)
        if date <= self.all_dates[-1]:
            raise ValueError('Proposed date must come after all dates in the '
                             'current `self.all_dates` array.')
        elif date.isoweekday() >= 6:
            raise ValueError('Proposed date must be a weekday.')

        # ensure that a simulation hasn't already been run
        if self.today != self.open_date:
            raise IndexError('You may only run `append_date()`*before* running '
                             'a simulation.')

        # ensure that all assets are present in price_dict
        if set(price_dict.keys()) != set(self.assets.keys()):
            raise ValueError('Differing assets in proposed dict and '
                             '`self.assets`. Please include all assets.')
        elif len(price_dict.keys()) != len(self.assets.keys()):
            raise ValueError('Differing number of assets in proposed dict and '
                             '`self.assets`. Ensure that their lengths match.')

        # add new row to each asset's dataFrame of price data
        for tk, val in self.assets.items():
            # create new row at the original dataFrame's end
            val['df'].loc[date] = val['df'].iloc[-1]

            # insert new price data into the new row
            val['df'].loc[date, 'adjOpen'] = price_dict[tk]
            val['df'].loc[date, 'adjClose'] = price_dict[tk]

        # change class' end date
        self.end_date = date

        # include new date in the date arrays and the rebalance dataFrame
        self.all_dates, self.active_dates = self._get_date_arrays()
        self.rb_info = self._calc_rebalance_info(verbose)

        # re-create result tracking arrays
        self._create_tracking_arrays()

        # finally, re-create any needed parent class attributes
        self.refresh_parent()

    def _create_tracking_arrays(self):
        '''
        Called in self.__init__() or (if run) self.append_date().

        Creates arrays that track the main portfolio's value over time, as well
        as the values of its core/satellite components (if present), the
        benchmark portfoliio (if present), and of the portfolio's unspent cash.
        '''
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
            dt = pd.Timestamp(dt)
            tk = tick_info.iloc[i]['ticker']
            if dt > self.open_date:
                dt_str = dt.strftime('%Y-%m-%d')
                od_str = self.open_date.strftime('%Y-%m-%d')
                raise ValueError(f"{tk}'s start date of {dt_str} is later than "
                                 "your open date (start date minus window) "
                                 f"of {od_str}. Try a later start date, a "
                                 "decreased window, or a different ticker.")

        # are all assets still active by self.end_date?
        for i, dt in enumerate(tick_info['endDate']):
            dt = pd.Timestamp(dt)
            tk = tick_info.iloc[i]['ticker']
            if dt < self.end_date:
                dt_str = dt.strftime('%Y-%m-%d')
                ed_str = self.end_date.strftime('%Y-%m-%d')
                raise ValueError(f"{tk}'s end date of {dt_str} is earlier "
                                 f"than your chosen end date of {ed_str}. "
                                 'Try an earlier end date or choose a '
                                 'different ticker.')

    def _validate_assets_dict(self, Portfolio, verbose):
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

        verbose : boolean, required
            Whether or not to print output from Portfolio.check_assets().
        '''
        # add a standard benchmark portfolio if one wasn't provided
        if len([tk for tk, info in Portfolio.assets.items()
                if info['label'] == 'benchmark']) == 0:
            # ensure that the potential additions exist during the simulation
            # period and are not already listed as core/satellite assets
            # (a future structural change might allow multiply-labeled assets)
            bench_stock = bench_bond = None

            # the stock index will be an ETF/mutual fund tracking the S&P 500
            if (  self.open_date >= pd.Timestamp(1993, 1, 29)
                  and 'SPY' not in Portfolio.assets.keys()  ):
                bench_stock = 'SPY'
            elif (  self.open_date >= pd.Timestamp(1976, 8, 31)
                    and 'VFINX' not in Portfolio.assets.keys()  ):
                bench_stock = 'VFINX'

            # the bond index will be an ETF/mutual fund tracking the
            # Barclays US Aggregate Bonds Index
            if (  self.open_date >= pd.Timestamp(2003, 12, 31)
                  and 'AGG' not in Portfolio.assets.keys()  ):
                bench_bond = 'AGG'
            elif (  self.open_date >= pd.Timestamp(1986, 12, 31)
                    and 'VBMFX' not in Portfolio.assets.keys()  ):
                bench_bond = 'VBMFX'

            # if valid stock & bond tickers were found, add the portfolio
            # (following the popular 60% stock/40% bond allocation model)
            if bench_stock is not None and bench_bond is not None:
                Portfolio.add_ticker(bench_stock, .6, label='benchmark')
                Portfolio.add_ticker(bench_bond, .4, label='benchmark')
            # else, the benchmark portfolio remains empty

        # run Portfolio's own validation function to be thorough
        Portfolio.check_assets(verbose)

        # verify that all assets are present over the user's entire date range
        self._verify_dates(Portfolio.tick_info)

        # if those tests pass, fetch historical data from online for each asset
        assets = pickle.loads(pickle.dumps(Portfolio.assets, -1))
        # (faster than copy.deepcopy for this use case)

        for tick, info in assets.items():
            # get daily open/close data from Tiingo
            df = self.call_tiingo(tick, self.open_date, self.end_date, verbose)

            # add the dataframe to the ticker's dictionary information
            info['df'] =  df

        # ensure that each asset has the same number of dates
        num_dates = np.unique([len(assets[nm]['df'].index) for nm in assets])
        assert len(num_dates) == 1, 'some ticker DataFrames are missing dates'

        return assets

    def _get_date_arrays(self):
        '''
        Called in __init__() of HistoricalSimulator.

        Traverses downloaded historical data and returns an array with all
        available dates (self.all_dates) and another only containing dates used
        in an eventual simulation (self.active_dates).

        Also changes self.start_date to the next market day if the user's
        original choice is absent from the data.
        '''
        # pick a ticker (shouldn't matter which; all should have same dates)
        nm = list(self.assets.keys())[0]
        df = self.assets[nm]['df']

        # save one array with all dates and another from start_date onward
        all_dates = df.index.copy()
        active_dates = df.loc[self.start_date:].index.copy()

        # change start/end dates to match data if either is absent from the data
        real_start = active_dates[0]
        real_end = active_dates[-1]
        fmt_str = '%Y-%m-%d'

        if (real_start.strftime(fmt_str) != self.start_date.strftime(fmt_str)):
            self.start_date = real_start
        if (real_end.strftime(fmt_str) != self.end_date.strftime(fmt_str)):
            self.end_date = real_end

        return all_dates, active_dates

    def _modify_rb_vars(self, rb_dates, sat_only, day, is_sat_only_rb=False):
        '''
        Called in self._calc_rebalance_info().

        `rb_dates` and `sat_only` can be two lists or a NoneType object and a
        pandas Series, respectively. This method handles that ambiguity by first
        trying to append `ind` to `obj` (i.e., the case where `rb_dates` and
        `sat_only` grow one item at a time because self.sat_rb_freq == 365.25.).

        If append() causes an AttributeError, the method pivots to flipping
        `sat_only`'s value at date `day`, since it must be a Series instead.
        (This is the case it is pre-filled with Trues and flips one to False.)

        See self._calc_rebalance_info() for more on how self.rb_info is built.

        Arguments
        ---------

        rb_dates : list or None, required
        sat_only : list or `pandas.Series`, required
            Types depend on this class instance's satellite rebalance frequency.

        day : `pandas.Timestamp` or `datetime.datetime`, required
            The date to be appended to `rb_dates`/`sat_only` or flipped in
            `sat_only`, depending on those arguments' types.

        is_sat_only_rb: boolean, optional
            The type of rebalance that will happen on date `day` in each asset's
            historical DataFrame. If True, it's a satellite-only rebalance;
            if False, it's for the total portfolio. [default: False]
        '''
        # need to check if day is None (final month scenario)
        if day is not None:
            try:
                rb_dates.append(day)
                sat_only.append(is_sat_only_rb)
            except AttributeError:
                sat_only.loc[day] = is_sat_only_rb
                # (no change needed with rb_dates)

        return rb_dates, sat_only

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
        next_month = pd.Timestamp(year, month, 28) + timedelta(days=4)
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
            ref_day = pd.Timestamp(yr, mth, 1)
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
                    date_range.append(dt)
                elif days_beyond_ref == rb_day + buffer:
                    date_range.append(dt)
                    break
                # iterate the loop's "days beyond rb_day" counter
                days_beyond_ref += 1

            # iterate the loop's date
            dt += timedelta(days=iter_day)

        return date_range

    def _calc_rebalance_info(self, verbose):
        '''
        Called in __init__() of HistoricalSimulator.

        Uses satellite and total portfolio rebalance frequencies to create
        `self.rb_info`, a dataFrame of this instance's rebalance dates and their
        types (satellite-only if True, total portfolio if False).

        Arguments
        ---------

        verbose : boolean, required
            Whether or not to print rebalance month information.
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)

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
            sat_mths = sat_mths[~np.isin(sat_mths, tot_mths)]
            my_pr('sat rb mths:', sat_mths, '\ntot rb mths:', tot_mths)

            # create list of dates when rebalances occur...
            # and another specifying which type -- satellite or total
            rb_dates = []
            sat_only = []

        else: # if daily... (only. in future could do every 2, 3 days and so on)
            # ...then every month has rebalance events
            sat_mths = all_months
            my_pr('sat rb mths:', sat_mths, '\ntot rb mths:', tot_mths)

            # include every active_date as a possible rebalance date
            # (total rebalance days will be flipped to True in sat_only later)
            rb_dates = None
            sat_only = pd.Series(index=self.active_dates,
                                 data=np.ones(len(self.active_dates),
                                              dtype=bool))

        #go = time.time()
        my_pr('all sim mths:')
        yr = self.start_date.year
        while yr <= self.end_date.year:
            # make array with all eligible months
            months = np.arange(1 if yr != self.start_date.year
                               else self.start_date.month,
                               13 if yr != self.end_date.year
                               else self.end_date.month + 1)
            # end month may not reach a reblance date, but allow for it if so

            my_pr(months, yr)
            # limit months to those cleared for rebalance events
            eligible = [mth for mth in months
                        if mth in tot_mths or mth in sat_mths]

            # check every month in the current year...
            for mth in eligible:
                # automatically make start_date a total rebalance event
                if (yr == self.start_date.year
                    and mth == self.start_date.month):
                    (rb_dates,
                     sat_only) = self._modify_rb_vars(rb_dates, sat_only,
                                                      self.start_date,
                                                      is_sat_only_rb=False)
                # in subsequent months, find desired market day for rebalancing
                else:
                    # get first and last possible rebalance days (using a range
                    # instead of a specific day for protection against holidays)
                    fnl = self._get_mth_rb_range(yr, mth)

                    # save dates that fall within that range
                    poss = self.active_dates[(min(fnl) <= self.active_dates)
                                             & (self.active_dates <= max(fnl))]

                    # save last/first day in the range as this month's rb date
                    try:
                        day = poss[-1 if self._target_rb_day < 0 else 0]
                    # if there are no dates in that range, use None instead
                    # (i.e., active_dates' last month cuts off prior to rb date)
                    except IndexError:
                        day = None
                        # NOTE: depending on buffer size in _get_mth_rb_range(),
                        # rb's could be triggered if sim ends within buffer but
                        # before target date. not yet sure how to fix this...

                    # update class' rb objects with this month's info
                    kind = True if mth not in tot_mths else False
                    (rb_dates,
                     sat_only) = self._modify_rb_vars(rb_dates, sat_only, day,
                                                      is_sat_only_rb=kind)

            yr += 1
        #my_pr(f"{time.time() - go:.3f} s for rebalance info loop")

        # make dataFrame of rebalance info with index of active dates
        rb_info = pd.DataFrame({'sat_only': sat_only})
        if type(rb_info.index) != pd.DatetimeIndex:
            rb_info.set_index(pd.DatetimeIndex(rb_dates), inplace=True)

        return rb_info

    def _get_static_rb_changes(self, names, main_portfolio=True):
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

        main_portfolio : boolean, optional
            If True, the method finds share changes for the main strategy's
            core/satellite portfolio. If False, the method finds share changes
            for the benchmark portfolio. [default: True]
        '''
        # get total value for portfolio in question
        pf_val = self.portfolio_value(self.today, main_portfolio=main_portfolio,
                                      at_close=False)

        # get share changes for assets in `names`
        deltas = []
        for name in names:
            ideal_frac = self.assets[name]['fraction']
            ideal_shares = pf_val * ideal_frac

            curr_price = self.assets[name]['df'].loc[self.today, 'adjOpen']
            curr_shares = self.assets[name]['shares']
            curr_held = curr_shares * curr_price

            # delta_shares must be an integer, so a full asset liquidation is
            # assumed any time it is within 1 of curr_shares (e.g., 87 & 87.74)
            delta_shares = (ideal_shares - curr_held) // curr_price
            delta_shares = (-curr_shares if curr_shares != 0
                            and curr_shares+delta_shares < 1 else delta_shares)
            deltas.append(delta_shares)

        return deltas

    def make_rb_trades(self, names, deltas, main_portfolio=True, verbose=False):
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
        prices = np.array([self.assets[nm]['df'].loc[self.today, 'adjOpen']
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

    def rebalance_portfolio(self, day, verbose=False):
        '''
        Called in self.begin_time_loop().

        General method that performs a total-portfolio rebalance by
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

        day : `pandas.Timestamp` or `datetime.datetime`, required
            The simulation's current date. Used to find whether this rebalance
            is satellite-only or for the total portfolio.

        verbose : boolean, optional
            If True, the method prints information on completed transactions.
            [default: False]
        '''
        my_pr = lambda *args, **kwargs: (print(*args, **kwargs)
                                         if verbose else None)

        my_pr(f"total rb; sat_only is {self.rb_info.loc[day, 'sat_only']}; "
              f"${self.cash:.2f} in account")

        # get share changes for core assets
        deltas = self._get_static_rb_changes(self.core_names)

        # get share changes for satellite assets from child's method
        deltas.extend(self.rebalance_satellite(day, verbose=verbose))
        deltas = np.array(deltas)
        my_pr('deltas:', deltas)

        # rebalance the main (core/satellite) strategy's portfolio
        main_names = np.array(self.core_names + self.sat_names)
        self.make_rb_trades(main_names, deltas, verbose=verbose)

        # next, get share changes for benchmark assets
        bench_deltas = self._get_static_rb_changes(self.bench_names,
                                                   main_portfolio=False)
        bench_deltas = np.array(bench_deltas)

        # rebalance the benchmark portfolio (no printed output for now)
        bench_names = np.array(self.bench_names)
        self.make_rb_trades(bench_names, bench_deltas, main_portfolio=False)

    def _check_dividends(self, main_portfolio=True, verbose=False):
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
            div_cash = self.assets[tk]['df'].loc[self.today, 'divCash']
            if div_cash == 0:
                continue

            # if this ticker isn't currently in the portfolio, skip to the next
            shares_held = self.assets[tk]['shares']
            if shares_held == 0:
                continue

            # barring those, receive the dividend as partial shares or cash
            if self.reinvest_dividends:
                tk_price = self.assets[tk]['df'].loc[self.today, 'adjOpen']
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
        WOULD HAVE TO RE-RUN _calc_rebalance_info() IN HERE AND REMAKE self.rb_info.

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
        go = time.time()
        for today in self.all_dates:
            self.today = today
            # "PRE-OPEN": update daily indicators based on YESTERDAY's CLOSES
            if self.today >= self.start_date:
                self.on_new_day()

            # "PRE-OPEN": cash in dividends from ex-date (YESTERDAY's) holdings,
            # once for main portfolio and once for benchmark
            if self.today >= self.start_date:
                self._check_dividends(verbose=verbose)
                self._check_dividends(main_portfolio=False, verbose=False)

            # AT OPEN: rebalance if needed
            if self.today in self.rb_info.index: # 2x faster than check by index
                # make rebalance calculations based on YESTERDAY'S STATS

                try:
                    my_pr('**** on', self.today.strftime('%Y-%m-%d'),
                          '\nvol streak: ', self.vol_streak,
                          'can enter:', self.can_enter,
                          'days out:', self.days_out)
                except AttributeError:
                    my_pr('**** on', self.today.strftime('%Y-%m-%d'))

                if self.rb_info.loc[self.today, 'sat_only'] == True: # sat-only
                    self.rebalance_satellite(self.today, verbose=verbose)
                else: # total portfolio
                    self.rebalance_portfolio(self.today, verbose=verbose)

            # AT CLOSE: track stats
            if self.today >= self.start_date:
                # save the main, core/satellite portfolio's value at day's close
                pf_value = self.portfolio_value(at_close=False)
                to_strategy_results.append(pf_value)

                # save the benchmark portfolio's value at day's close
                bench_pf_value = self.portfolio_value(main_portfolio=False,
                                                      at_close=False)
                to_bench_results.append(bench_pf_value)

                # save amount of free cash left in the main portfolio
                to_cash_over_time.append(self.cash)

                # tried assigning to DataFrames with .loc on advice from
                # https://stackoverflow.com/a/45983830
                # turns out it's the fastest pandas assignment method, but list
                # append and numpy array assignment are 200x faster in this case

            # save today's date before moving to next iteration (for potential
            # use in getting previous close in on_new_day() post-start_date)
            self.prev_day = self.today

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
        my_pr('end date main portfolio value: '
              f"${self.portfolio_value(self.end_date):,.02f}")

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
                  f"${self.portfolio_value(self.end_date, main_portfolio=False):,.02f}")

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
            tick_info = self.assets[tk]['df'].loc[self.start_date:,
                                                  'adjClose'].copy()
            tick_info *= start_value / tick_info.iloc[0]
            results.append(tick_info.values)

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
            tick_info.plot(label=f"{tk} ({tick_type})",
                           lw=1.5, c=col,
                           fontsize=14, logy=logy, ax=ax)

            # print final value of ticker holdings
            my_pr(f"{tk} ending value: ${tick_info.iloc[-1]:,.2f}")

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
