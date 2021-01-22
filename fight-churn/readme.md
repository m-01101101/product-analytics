# Fight churn with data

This repo is based on the excellent book "fight churn with data" [website](https://fightchurnwithdata.com/), [git repo](https://github.com/carl24k/fight-churn)

<img src="md_refs/covertitle.png" width=200>

_Do note, I have not copied the repo exactly and have made a few changes to the code_. 

## Thinking about customer behaviour

The data is created through a simulation. A `Customer` class is created, each customer performs a number of behaviours at varied frequencies. Dependent on their use of the product and their propensity to be satisfied `customer.satisfaction_propensity` they will get some level of "utility" from the product. The higher their utility, the lower the likelihood of churn.

Though this all sounds a bit "economic man", something that couldn't possibly reflect reality, the result is behaviour that looks pretty realistic. On an aggregate level it does a good job and providing data that will look familiar out in the wild.

It is worth investigating the `customer`, `behavior` and `utility` modules to understand the underlying assumptions of the "people" being modelled in more detail.
