# Understanding the data

## Generate

Generate the data to a local postgres database.

The defaults for can altered in these two scripts.

```zsh
# create the schema and relevant table
>>> python data-generation/py/churndb.py

Creating schema socialnet7 (if not exists)...
Creating table event (if not exists)
Creating table subscription (if not exists)
Creating table event_type (if not exists)
Creating table metric (if not exists)
Creating table metric_name (if not exists)
Creating table active_period (if not exists)
Creating table observation (if not exists)
Creating table active_week (if not exists)
Creating table account (if not exists)

# simulate customer behaviour
# default 6 months, 10,000, 10% growth rate
>>> python data-generation/py/churnsim.py

Created 20000 initial customers with 135109 subscriptions for start date 2020-01-01
...
```

## EDA

We'll now have `event` data and `subscription` data.

Summary of event data;

```SQL
with base as (
  select 
      event_type_name
      , count(*) event_count
      , count(distinct account_id) unique_users
  from socialnet7.event
  join socialnet7.event_type using(event_type_id)
  group by 1
)

select *
    , round(event_count*1.0 / unique_users, 3) avg_time_performed_per_user 
from base
order by 2 desc
```

TODO create markdown table or rich print output of query

Summary of subscription data;

```SQL
with base as (
  select 
      account_id
      , count(*) subscriptions
      , min(start_date) first_subscription
      , max(start_date) last_subscription
  from socialnet7.subscription
  group by 1
)

select 
    subscriptions
    , round(count(account_id) over(partition by subscriptions)*1.0
        / count(account_id) over()
        , 3) pc
from base
```
