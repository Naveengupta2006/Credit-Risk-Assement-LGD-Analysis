-- Active: 1768293064226@@localhost@3306@practice
select * from loan

-- rank highest risk customer
select customer_name, LGD,
rank() over(order by LGD desc) as risk_rank
from loan
where default_date is not null

-- rank highest loan exposure

select customer_name, sum(loan_amount) as total_exposure,
rank() over(order by sum(loan_amount) desc) as exposure_rank
from loan
group by customer_name