-- Active: 1768293064226@@localhost@3306@practice
select * from loan

-- average recovery per loan type
select loan_type, avg(1 - recovery_rate) as recovery_rate
from loan
where default_date is not null
group by loan_type;

-- highest LGD accounts
select customer_name, loan_amount, LGD
from loan
where default_date is not NULL
order by LGD DESC
limit 10;

-- collateral shortfall cases
select customer_name, loan_amount, collateral_value
from loan
where loan_amount > collateral_value

