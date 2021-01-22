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
order by 2 desc;