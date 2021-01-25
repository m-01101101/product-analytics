with base as (
    select *
    , (DATE_PART('month', start_date::date) - DATE_PART('month', previous_month::date)) month_diff
    from (
    select *
        , lag(start_date, 1) over(partition by account_id order by start_date) previous_month
    from socialnet7.subscription
    where start_date != '2020-09-01'  -- don't include most recent month
    ) subs
)

, retained_revenue as (
    select
        date_trunc('month', start_date)::date as month_ds
        , count(distinct account_id) retained_subs
        , sum(mrr) retained_revenue
    from base
    where month_diff = 1 
    group by 1
)  

, previous_month_revenue as (
    select
        (date_trunc('month', start_date)::date + interval'1 month')::date as month_ds
        , count(distinct account_id) previous_month_subs
        , sum(mrr) previous_month_revenue
    from socialnet7.subscription
    group by 1
)

select *
	, retained_subs / previous_month_subs as net_monthly_sub_retention_rate
    , 1.0 - (retained_subs / previous_month_subs) as net_monthly_sub_churn_rate
	, retained_revenue / previous_month_revenue as net_mrr_retention_rate
    , 1.0 - (retained_revenue / previous_month_revenue) as net_mrr_churn_rate
from retained_revenue
join previous_month_revenue using(month_ds)
order by month_ds
