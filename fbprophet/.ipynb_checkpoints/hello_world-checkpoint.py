"""
playing around with fbprophet
"""
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbprophet import Prophet
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from rich import print


columns = ["created_at", "id"]
tic = dt.datetime.now()
_df = pd.read_csv("../cleo_example/data/cleo_users.csv", usecols=columns, low_memory=False)
print(f"""time to read csv: {str(dt.datetime.now() - tic)}\n
shape: {_df.shape}
""")

_df_tidy = _df.copy()
_df_tidy = _df_tidy.drop_duplicates(keep='last')
_df_tidy["date"] = pd.to_datetime(_df_tidy.created_at, format='%Y-%m-%d')

assert _df_tidy.id.nunique() == _df_tidy.shape[0]

df = _df_tidy.groupby("date").agg(
    daily_sign_ups=("id", "nunique")
).reset_index()

df.sort_values(by="date", inplace=True)
# df.set_index('date', inplace=True)
# df.sort_index(inplace=True)

print(f"working dataset for prohpet:\n{df.head()}\n")

def plot_line(df: pd.DataFrame, x_axis: str, line: str, c: str = "b"):
    x = df[x_axis]
    line = df[line]

    fig = plt.figure(dpi=100, figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.yaxis.grid()

    ax.plot(x, line, color=c)
    title = line.replace("_", " ")
    ax.set_title(f"{title}")
    ax.set_ylabel(f"{line}")
    ax.set_xlabel(f"{x}")
    # ax.set_xticklabels(x, rotation=-40)

    plt.show();
    return

## 
plot_line(df, df.columns[0], df.columns[1])


"""
box-Cox transforms are data transformations that evaluate a set of lambda coefficients (λ) 
and selects the value that achieves the best approximation of normality

the boxcox method returns a positive dataset transformed by a Box-Cox power transformation
the boxcox method has one required input: a 1-dimensional array of positive data to transform

you can also specify the λ value you’d like to use for your transformation (e.g. λ = 0 for a log transform)
otherwise, the boxcox method will find the λ that maximizes the log-likelihood function 
and will return it as the second output argument
"""

# Apply Box-Cox Transform to value column and assign to new column y
df['y'], lam = boxcox(df.daily_sign_ups)

plot_line(df, "date", "y", "green")

# instantiating (create an instance of) a Prophet object
m = Prophet()

# must be ds not date
df.rename(columns={"date": "ds"}, inplace=True)
m.fit(df[["ds", "y"]])

"""
Prophet will create a new dataframe assigned to the forecast variable 
that contains the forecasted values for future dates under the column yhat,
as well as uncertainty intervals and components for the forecast.
"""
future = m.make_future_dataframe(periods=183)
forecast = m.predict(future)

print(f"\nplotting fbprobhet forecast for 6 months\n")
m.plot(forecast)
print(f"\nplotting fbprobhet components for 6 months\n")
m.plot_components(forecast)


"""
since Prophet was used on the Box-Cox transformed data, 
you'll need to transform your forecasted values back to their original units

the inv_boxcox method has two required inputs; 
    an array of data to transform
    a λ value for the transform

we have the λ value from in the "lam variable" from our Box-Cox transformation
"""

# transformaing forecasted values back to their original units
forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))
print(f"\nplotting fbprobhet forecast for 6 months in original units\n")
m.plot(forecast)
