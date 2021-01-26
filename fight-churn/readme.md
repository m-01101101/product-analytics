# Fight churn with data

This repo is based on the excellent book "fight churn with data" [website](https://fightchurnwithdata.com/), [git repo](https://github.com/carl24k/fight-churn)

<img src="md_refs/covertitle.png" width=200>

_Do note, I have not copied the repo but rather built my own from scratch using a similar simulation and framework_.

## Thinking about customer behaviour

The data is created through a simulation. A `Customer` class is created, each customer performs a number of behaviours at varied frequencies. Dependent on their use of the product and their propensity to be satisfied `customer.satisfaction_propensity` they will get some level of "utility" from the product. The higher their utility, the lower the likelihood of churn.

Though this all sounds a bit "economic man", something that couldn't possibly reflect reality, the result is behaviour that looks pretty realistic. On an aggregate level it does a good job and providing data that will look familiar out in the wild.

It is worth investigating the `customer`, `behavior` and `utility` modules to understand the underlying assumptions of the "people" being modelled in more detail.

## Defining retention and churn

Both Carl and [Johnathan Hsu](https://tribecap.co/a-quantitative-approach-to-product-market-fit/) focus on describing retention first in terms of revenue. This takes into account not only customers churning but the amount of revenue you get from existing customers (ie are retained customers spending more, or upgrading to premium plans?). Up-selling existing customers (increasing incremental revenue) can therefore make churn seem lower than it is as a user level.

The net retention rate (NRR) is the proportion of revenue received at the end of the period from existing customers only

> Net retention rate $NRR$ = (retained(t) + expansion(t) - churned(t) - contraction(t)) / revenue(t-1)
>
> Net churn = 1 - NRR

Gross retention does not include the filter of only using customers present in the previous time period. It looks only at the aggregate revenue numbers;

> Gross retention = revenue(t) / revenue(t-1)
>
> Gross churn = 1 - gross retention rate

It's beneficial to calculate these same measures but counting customers rather than summing revenue. This will cancel out the expansion or contraction (up-sells / down-sells) of retained customers.

If there is wide variation in the price customers pay (in B2B different tiers may be extremely different prices), or the revenue is heavily skewed (some people pay a lot, most very little), then it's useful to apply a log transformation to the revenue numbers to decrease the noise in `expansion` and `contraction` of retained users.
