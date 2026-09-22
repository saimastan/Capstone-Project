
# Zepto Data & AI Platform — Module 1

## 1. Project Overview

Module 1 focuses on building a data pipeline using web scraping,
data cleaning, SQLite database design, SQL queries, and Pandas.

The pipeline follows:

Scraping → Cleaning → Currency Conversion → SQLite → SQL → Pandas

---

## 2. Data Source

Data was collected from:

Books to Scrape
https://books.toscrape.com/

The website is used for scraping practice and does not require
login credentials or an API key.

---

## 3. Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- Google Colab

---

## 4. Data Collection

The scraper automatically collected book information from
multiple categories.

The following fields were collected:

- Title
- Price
- Star Rating
- Availability
- Category

At least 60 books were collected from at least 3 categories.

---

## 5. Data Cleaning

The scraped data was cleaned before storing it in SQLite.

### Price

The original price was converted from text to float.

Example:

£51.77 → 51.77

### Rating

Star ratings were converted to integers:

One → 1
Two → 2
Three → 3
Four → 4
Five → 5

### Availability

Availability was converted into a Boolean value:

In stock → True
Not in stock → False

---

## 6. Currency Conversion

A fixed conversion rate was used:

1 GBP = 105.50 INR

Formula:

price_inr = price_gbp × 105.50

---

## 7. SQLite Database

The cleaned data was stored in SQLite.

Two related tables were created:

### categories

- category_id
- category_name

### books

- book_id
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id

The `category_id` connects the books table with the categories table.

---

## 8. SQL Queries

The project contains SQL queries demonstrating:

1. SELECT and WHERE
2. ORDER BY
3. LIMIT
4. DISTINCT
5. BETWEEN
6. JOIN

The SQL queries were executed using SQLite and their outputs
were displayed in the notebook.

---

## 9. Pandas read_sql()

SQL query results were loaded into Pandas DataFrames using:

pd.read_sql()

At least two SQL query results were loaded into Pandas.

---

## 10. Pandas Merge

The relationship between books and categories was reproduced
using:

pd.merge()

The Pandas merge result was compared with the SQL JOIN result
to verify that both approaches produced equivalent results.

---

## 11. Project Workflow

Scraping
↓
Cleaning
↓
Currency Conversion
↓
SQLite Database
↓
SQL Queries
↓
Pandas read_sql
↓
Pandas merge

---

## 12. Conclusion

Module 1 demonstrates an end-to-end data pipeline starting from
web scraping and ending with SQL and Pandas analysis.

The pipeline provides the data foundation required for the
next modules of the Zepto Data & AI Platform.
