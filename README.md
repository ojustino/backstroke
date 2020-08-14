## `backstroke` 💧
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ojustino/backstroke/master?filepath=walkthrough.ipynb)
<br>
*( ^ click above to run in the cloud)*

A Python package that allows users to create portfolio Strategy classes based
on a "safe core, risky satellite" model and simulate their performances on
historical stock data from [Tiingo's free API](https://api.tiingo.com/).

I created this repository because I wanted to research the purported dangers of
leveraged ETFs for myself. I tried sites like [Portfolio Visualizer](
https://www.portfoliovisualizer.com/) and [QuantConnect](
https://quantconnect.com/), but ended up making my own Python package to give
myself more flexibility in creating strategies than the latter allowed *and* 
more visualization options than the former.

Now, I actually use a Strategy class to make decisions for my Roth IRA.
(_Naturally, your results may vary._)

**Skills used:**
<br>
_(bear with me; I'm job-hunting)_

data(Frame) manipulation with `pandas`, fetching data over HTTP with `requests`,
object-oriented programming with abstract base classes, visualization with
`matplotlib`, cloud-based Jupyter environment creation with Binder and Docker.

### Example usage:

Read through
[`walkthrough.ipynb`](https://github.com/ojustino/tennis-abs-api/blob/master/walkthrough.ipynb)
and [`buy_and_hold.ipynb`](https://github.com/ojustino/tennis-abs-api/blob/master/walkthrough.ipynb)
for a quick introduction.

### Installation ***(coming soon)***:

```
git clone https://github.com/ojustino/backstroke
cd backstroke
pip install .
```
(Add `-e` before the period in the final line if you intend to make changes to the source code.)

### License:

This project uses a slightly modified version of the PolyForm Noncommercial
License 1.0.0. Basically, you're free to view, run, download, and modify this
package for any non-commercial purpose. For more details, read the license in
full [here](https://github.com/ojustino/tennis-abs-api/blob/master/LICENSE.md).
