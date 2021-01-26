with base as (
  select
  	cohort_base.*
  	, (date_part('month', active_month) - date_part('month', cohort_month)) cohort_period
  from (
    select distinct
      account_id
      , date_trunc('month', min(event_time) over(partition by account_id))::date as cohort_month
      , date_trunc('month', event_time)::date active_month
    from socialnet7.event
    ) cohort_base
)

, cohort_size as (
  select
  	cohort_month
  	, count(distinct account_id) users
  from base
  group by 1
)

, activity as (
  select
  	cohort_month
  	, cohort_period
  	, count(distinct account_id) mau
  from base 
  group by 1, 2
)

select
	cohort_month
    , cohort_period
    , cohort_size.users as cohort_size
    , activity.mau
    , activity.mau::int*1.0 / cohort_size.users::int as retention_pc
from activity
join cohort_size using(cohort_month)
where cohort_month != '2020-09-01'
order by 1, 2
