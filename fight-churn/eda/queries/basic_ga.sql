with base as (
    select *
  	, case when previous_month is NULL then 1 else 0 end is_acquired
  	, case 
  		when (DATE_PART('month', start_date::date) - DATE_PART('month', previous_month::date)) = 1 
  		then 1 else 0 
 	end is_retained
  	, case
  		when next_month is NULL and end_date < '2020-09-01' 
  		then 1 else 0
 	end is_churned
    from (
        select *
            , lag(start_date, 1) over(partition by account_id order by start_date) previous_month
            , lead(start_date, 1) over(partition by account_id order by start_date) next_month
        from socialnet7.subscription
    ) subs
)

, aggregation as (
select 
	date_trunc('month', start_date)::date month_ds
    , sum(is_acquired) acquired_users
    , sum(is_retained) retained_users
    , sum(is_churned)*-1 churned_users
from base
where date_trunc('month', start_date)::date < '2020-09-01'
group by 1
)

select *
    , round((acquired_users + retained_users)*1.0 / nullif(abs(churned_users), 0), 3) as quick_ratio
from aggregation
order by 1
