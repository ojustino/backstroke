## `backstroke` 💧
<p>
  <a href="https://mybinder.org/v2/gh/ojustino/backstroke/master?filepath=walkthrough.ipynb" target="_blank">
    <img src="https://mybinder.org/badge_logo.svg"/>
  </a>
  <!-- <a href="https://travis-ci.com/github/ojustino/backstroke/" target="_blank">
    <img src="https://travis-ci.org/ojustino/backstroke.svg?branch=master"/>
  </a> -->
  <br />
  <i> ( ⬆️ click above to run in the cloud) </i>
</p>

A Python package that allows users to create portfolio Strategy classes based
on a <a href="https://www.investopedia.com/articles/financial-theory/08/core-satellite-investing.asp" target="_blank">
    "safe core, risky satellite"
</a> model and simulate their performances on historical stock data from
<a href="https://api.tiingo.com/" target="_blank">a free API</a>
hosted by Tiingo.

I created this repository because I wanted to research
<a href="https://www.investopedia.com/articles/financial-advisors/082515/why-leveraged-etfs-are-not-longterm-bet.asp" target="_blank">
    the purported dangers of leveraged ETFs
</a> (which magnify an index's daily gains and losses) for myself. I tried sites
like <a href="https://quantconnect.com/" target="_blank">QuantConnect</a> and
<a href="https://www.portfoliovisualizer.com/" target="_blank">
    Portfolio Visualizer
</a> but resolved to write my own Python package to gain more flexibility in
creating strategies than the former allows *and* more visualization options
than exist in the latter.

Now, I actually use a `Strategy` class to make decisions for my IRA.
(_Naturally, your results may vary._)

**Skills used:**
<br>
_(bear with me; I'm job-hunting)_

data(Frame) manipulation with `pandas`, fetching data over HTTP with `requests`,
object-oriented programming with abstract base classes, visualization with
`matplotlib`, cloud-based Jupyter environment creation with Binder and Docker.

### Example usage:

Read through
<a href="https://github.com/ojustino/backstroke/blob/master/walkthrough.ipynb" target="_blank">
    `walkthrough.ipynb`
</a> and
<a href="https://github.com/ojustino/backstroke/blob/master/buy_and_hold.ipynb" target="_blank">
    `buy_and_hold.ipynb`
</a> to get familiar with the package. Or, click the badge atop this file for an
interactive walkthrough.

### Installation ***(coming soon)***:

```
git clone https://github.com/ojustino/backstroke
cd backstroke
pip install .
```
(Add `-e` before the period in the final line if you intend to make changes to the source code.)

### License:

This project uses a
<a href="https://github.com/ojustino/backstroke/blob/master/LICENSE.md" target="_blank">
    slightly modified version
<a/> of the PolyForm Noncommercial License 1.0.0. Basically, you're free to
view, run, download, and modify this package for any non-commercial purpose.
