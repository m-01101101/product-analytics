"""
for each event in event_type, we want to calculate these metrics

ref https://github.com/carl24k/fight-churn/blob/master/listings/py/run_churn_listing.py

48 min https://www.youtube.com/watch?v=XvuHDBJ_rX8&ab_channel=FightingChurnWithDataScience
"""

sql = """with date_vals AS (
 	select i::timestamp as metric_date 
from generate_series('%from_yyyy-mm-dd', '%to_yyyy-mm-dd', '7 day'::interval) i
)

insert into metric (account_id, metric_time, metric_name_id, metric_value)

select 
    account_id
    , metric_date
    , %new_metric_id
    , count(*) as metric_value
from socialnet7.event e 
inner join date_vals d
    on e.event_time < metric_date 
    and e.event_time >= metric_date - interval '28 day'
inner join socialnet7.event_type t 
    on t.event_type_id=e.event_type_id
where t.event_type_name='%event2measure'
group by account_id, metric_date
ON CONFLICT DO NOTHING;
"""