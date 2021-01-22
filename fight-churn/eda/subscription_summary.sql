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