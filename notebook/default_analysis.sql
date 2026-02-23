-- Active: 1768293064226@@localhost@3306@practice
select * from loan

-- total number of default
select count(*) as total_default
from main_loan
where default_date is not NULL

-- default rate by loan type
select loan_type,
    sum(case when default_date is not null then 1 else 0 end) / count(*) as default_rate
from loan
GROUP BY loan_type

-- default rate by tenure
select tenure_years,
    sum(case when default_date is not null then 1 else 0 end) / count(*) as default_rate
from loan
group by tenure_years

