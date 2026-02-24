-- Active: 1768293064226@@localhost@3306@practice

select * from loan

-- customers with more than 2 loans
select customer_name, count(loan_amount) as total_loan
from loan
group by customer_name
having count(loan_amount) > 1

-- top 10 highest loan amounts
select customer_name, loan_amount
from loan
order by loan_amount DESC
limit 10

-- total exposure per customer
select customer_name, sum(loan_amount) as total_exposure, count(loan_amount) as total_loan
from loan
group by customer_name