with date_vals AS (
  	select i::timestamp as metric_date 
	from generate_series('%from_yyyy-mm-dd', '%to_yyyy-mm-dd', '7 day'::interval) i
)
select 
    account_id
    , metric_date
    , count(*) as n_%event2measure
from socialnet7.event e 
inner join date_vals d
    on e.event_time < metric_date
    and e.event_time >= metric_date - interval '28 day'
inner join socialnet7.event_type t 
    on t.event_type_id=e.event_type_id
where t.event_type_name='%event2measure'
group by account_id, metric_date
order by account_id, metric_date;