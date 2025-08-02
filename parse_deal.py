import os, openai, ast, re
import pandas as pd
import numpy as np

from enum import Enum
from pydantic import BaseModel

# Load API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

CHAIR_BATNA_BUYER = 120
CHAIR_BATNA_SELLER = 40
TABLE_BATNA_BUYER = 200
TABLE_BATNA_SELLER = 100

exercise = 'table'  # Change to 'rental' or 'hfc' as needed
model = "gpt-4o-mini"
temperature = 0.2
round = "round1"
test = False

negotiation_file = f'negotiations/{round}_{exercise}_negotiations.csv'
outcomes_file = f'outcomes/{round}_{exercise}_outcomes.csv'

class TableDeal(BaseModel):
    deal_amount: float | None

def parse_deal_distributive(conversation_history, model, temperature):
    if 'NO DEAL' in conversation_history:
        return None
    DEAL_RE = re.compile(
        r"\$(\d+(?:\.\d+)?)(?![\s\S]*\$)",   # negative look-ahead: no more $ ahead
        flags=re.DOTALL                       # make '.' span new-lines
    )
    m = DEAL_RE.search(conversation_history)
    deal_amount = float(m.group(1)) if m else None
    return deal_amount

    # no_deal = 'NO DEAL' in conversation_history
    # if no_deal:
    #     return None
    # text = f"""
    # What was the amount of the deal in this conversation?
    
    # <conversation>
    # {conversation_history}
    # </conversation>

    # Respond ONLY with a number. Do not include any text.
    # """
    # response = openai.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content": text}],
    #     temperature=temperature,
    # )
    # deal_amount = response.choices[0].message.content
    # deal_amount = float(deal_amount.replace('$', '').replace(',', '').strip())
    return deal_amount

class RentAmount(Enum):
    Rent_3100_per_month = 3100
    Rent_3300_per_month = 3300
    Rent_3500_per_month = 3500
    Rent_3700_per_month = 3700
    Rent_3900_per_month = 3900

class Deposit(Enum):
    Deposit_500 = 500
    Deposit_1000 = 1000
    Deposit_1500 = 1500
    Deposit_2000 = 2000
    Deposit_2500 = 2500

class StartDate(Enum):
    May_1 = "May 1"
    May_15 = "May 15"
    June_1 = "June 1"
    June_15 = "June 15"
    July_1 = "July 1"

class ContractLength(Enum):
    Month_to_month = "Month-to-month"
    Three_months = "Three months"
    Six_months = "Six months"
    One_year = "One year"
    Two_years = "Two years"

class RentalDeal(BaseModel):
    rent_amount: RentAmount
    deposit: Deposit
    start_date: StartDate
    contract_length: ContractLength

def get_points_tenant(rental_deal):
    points = 0
    if rental_deal.rent_amount == RentAmount.Rent_3100_per_month:
        points += 1250
    elif rental_deal.rent_amount == RentAmount.Rent_3300_per_month:
        points += 1050
    elif rental_deal.rent_amount == RentAmount.Rent_3500_per_month:
        points += 850
    elif rental_deal.rent_amount == RentAmount.Rent_3700_per_month:
        points += 650
    elif rental_deal.rent_amount == RentAmount.Rent_3900_per_month:
        points += 450
    
    if rental_deal.deposit == Deposit.Deposit_500:
        points += 1250
    elif rental_deal.deposit == Deposit.Deposit_1000:
        points += 1050
    elif rental_deal.deposit == Deposit.Deposit_1500:
        points += 850
    elif rental_deal.deposit == Deposit.Deposit_2000:
        points += 650
    elif rental_deal.deposit == Deposit.Deposit_2500:
        points += 450
    
    if rental_deal.start_date == StartDate.May_1:
        points += 1250
    elif rental_deal.start_date == StartDate.May_15:
        points += 1050
    elif rental_deal.start_date == StartDate.June_1:
        points += 850
    elif rental_deal.start_date == StartDate.June_15:
        points += 650
    elif rental_deal.start_date == StartDate.July_1:
        points += 450
    
    if rental_deal.contract_length == ContractLength.Month_to_month:
        points += 1250
    elif rental_deal.contract_length == ContractLength.Three_months:
        points += 1050
    elif rental_deal.contract_length == ContractLength.Six_months:
        points += 850
    elif rental_deal.contract_length == ContractLength.One_year:
        points += 650
    elif rental_deal.contract_length == ContractLength.Two_years:
        points += 450
    
    return points

def get_points_landlord(rental_deal):
    points = 0
    if rental_deal.rent_amount == RentAmount.Rent_3100_per_month:
        points += 450
    elif rental_deal.rent_amount == RentAmount.Rent_3300_per_month:
        points += 650
    elif rental_deal.rent_amount == RentAmount.Rent_3500_per_month:
        points += 850
    elif rental_deal.rent_amount == RentAmount.Rent_3700_per_month:
        points += 1050
    elif rental_deal.rent_amount == RentAmount.Rent_3900_per_month:
        points += 1250
    
    if rental_deal.deposit == Deposit.Deposit_500:
        points += 450
    elif rental_deal.deposit == Deposit.Deposit_1000:
        points += 650
    elif rental_deal.deposit == Deposit.Deposit_1500:
        points += 850
    elif rental_deal.deposit == Deposit.Deposit_2000:
        points += 1050
    elif rental_deal.deposit == Deposit.Deposit_2500:
        points += 1250
    
    if rental_deal.start_date == StartDate.May_1:
        points += 450
    elif rental_deal.start_date == StartDate.May_15:
        points += 650
    elif rental_deal.start_date == StartDate.June_1:
        points += 850
    elif rental_deal.start_date == StartDate.June_15:
        points += 1050
    elif rental_deal.start_date == StartDate.July_1:
        points += 1250
    
    if rental_deal.contract_length == ContractLength.Month_to_month:
        points += 450
    elif rental_deal.contract_length == ContractLength.Three_months:
        points += 650
    elif rental_deal.contract_length == ContractLength.Six_months:
        points += 850
    elif rental_deal.contract_length == ContractLength.One_year:
        points += 1050
    elif rental_deal.contract_length == ContractLength.Two_years:
        points += 1250
    
    return points

def parse_deal_rental(conversation_history, model, temperature):
    # This would need to be implemented based on the rental negotiation format
    # For now, returning a placeholder
    return None

class LumpSumFee(Enum):
    Option_25000 = 25000
    Option_30000 = 30000
    Option_35000 = 35000
    Option_40000 = 40000
    Option_45000 = 45000

class DiscretionaryBudget(Enum):
    NoDiscretionaryBudget = 0
    DiscretionaryBudget_5000 = 5000
    DiscretionaryBudget_10000 = 10000
    DiscretionaryBudget_15000 = 15000
    DiscretionaryBudget_20000 = 20000

class TravelExpenses(Enum):
    Bus_or_train_fare_to_destinations_within_250_miles_otherwise_economy_class_airfare_anywhere_else = "Bus or train fare to destinations within 250 miles otherwise economy class airfare anywhere else"
    Economy_class_airfare_to_anywhere = "Economy class airfare to anywhere"
    Economy_class_airfare_within_the_United_States_otherwise_Business_Class_airfare_internationally = "Economy class airfare within the United States otherwise Business Class airfare internationally"
    Business_Class_airfare_within_the_United_States_First_Class_airfare_internationally = "Business Class airfare within the United States First Class airfare internationally"
    First_Class_airfare_anywhere = "First Class airfare anywhere"

class InvoiceFrequency(Enum):
    SentWeekly = "Invoices sent out weekly (every 7 days)"
    SentBiWeekly = "Invoices sent out bi-weekly (every 14 days)"
    SentMonthly = "Invoices sent out monthly (every 30 days)"
    SentSixWeeks = "Invoices sent out every 6 weeks (every 42 days)"
    OnceAtEndOfSummer = "Only one invoice at the end of the summer"

class ConsultantDeal(BaseModel):
    lump_sum_fee: LumpSumFee
    discretionary_budget: DiscretionaryBudget
    travel_expenses: TravelExpenses
    invoice_frequency: InvoiceFrequency

def get_points_consultant(consultant_deal):
    points = 0
    if consultant_deal.lump_sum_fee == LumpSumFee.Option_25000:
        points += 1250
    elif consultant_deal.lump_sum_fee == LumpSumFee.Option_30000:
        points += 1050
    elif consultant_deal.lump_sum_fee == LumpSumFee.Option_35000:
        points += 850
    elif consultant_deal.lump_sum_fee == LumpSumFee.Option_40000:
        points += 650
    elif consultant_deal.lump_sum_fee == LumpSumFee.Option_45000:
        points += 450
    
    if consultant_deal.discretionary_budget == DiscretionaryBudget.NoDiscretionaryBudget:
        points += 450
    elif consultant_deal.discretionary_budget == DiscretionaryBudget.DiscretionaryBudget_5000:
        points += 650
    elif consultant_deal.discretionary_budget == DiscretionaryBudget.DiscretionaryBudget_10000:
        points += 850
    elif consultant_deal.discretionary_budget == DiscretionaryBudget.DiscretionaryBudget_15000:
        points += 1050
    elif consultant_deal.discretionary_budget == DiscretionaryBudget.DiscretionaryBudget_20000:
        points += 1250
    
    if consultant_deal.travel_expenses == TravelExpenses.Bus_or_train_fare_to_destinations_within_250_miles_otherwise_economy_class_airfare_anywhere_else:
        points += 450
    elif consultant_deal.travel_expenses == TravelExpenses.Economy_class_airfare_to_anywhere:
        points += 650
    elif consultant_deal.travel_expenses == TravelExpenses.Economy_class_airfare_within_the_United_States_otherwise_Business_Class_airfare_internationally:
        points += 850
    elif consultant_deal.travel_expenses == TravelExpenses.Business_Class_airfare_within_the_United_States_First_Class_airfare_internationally:
        points += 1050
    elif consultant_deal.travel_expenses == TravelExpenses.First_Class_airfare_anywhere:
        points += 1250
    
    if consultant_deal.invoice_frequency == InvoiceFrequency.SentWeekly:
        points += 1250
    elif consultant_deal.invoice_frequency == InvoiceFrequency.SentBiWeekly:
        points += 1050
    elif consultant_deal.invoice_frequency == InvoiceFrequency.SentMonthly:
        points += 850
    elif consultant_deal.invoice_frequency == InvoiceFrequency.SentSixWeeks:
        points += 650
    elif consultant_deal.invoice_frequency == InvoiceFrequency.OnceAtEndOfSummer:
        points += 450
    
    return points

def get_points_coo(consultant_deal):
    points = 0
    if consultant_deal.lump_sum_fee == LumpSumFee.Option_25000:
        points += 450
    elif consultant_deal.lump_sum_fee == LumpSumFee.Option_30000:
        points += 650
    elif consultant_deal.lump_sum_fee == LumpSumFee.Option_35000:
        points += 850
    elif consultant_deal.lump_sum_fee == LumpSumFee.Option_40000:
        points += 1050
    elif consultant_deal.lump_sum_fee == LumpSumFee.Option_45000:
        points += 1250
    
    if consultant_deal.discretionary_budget == DiscretionaryBudget.NoDiscretionaryBudget:
        points += 1250
    elif consultant_deal.discretionary_budget == DiscretionaryBudget.DiscretionaryBudget_5000:
        points += 1050
    elif consultant_deal.discretionary_budget == DiscretionaryBudget.DiscretionaryBudget_10000:
        points += 850
    elif consultant_deal.discretionary_budget == DiscretionaryBudget.DiscretionaryBudget_15000:
        points += 650
    elif consultant_deal.discretionary_budget == DiscretionaryBudget.DiscretionaryBudget_20000:
        points += 450
    
    if consultant_deal.travel_expenses == TravelExpenses.Bus_or_train_fare_to_destinations_within_250_miles_otherwise_economy_class_airfare_anywhere_else:
        points += 1250
    elif consultant_deal.travel_expenses == TravelExpenses.Economy_class_airfare_to_anywhere:
        points += 1050
    elif consultant_deal.travel_expenses == TravelExpenses.Economy_class_airfare_within_the_United_States_otherwise_Business_Class_airfare_internationally:
        points += 850
    elif consultant_deal.travel_expenses == TravelExpenses.Business_Class_airfare_within_the_United_States_First_Class_airfare_internationally:
        points += 650
    elif consultant_deal.travel_expenses == TravelExpenses.First_Class_airfare_anywhere:
        points += 450
    
    if consultant_deal.invoice_frequency == InvoiceFrequency.SentWeekly:
        points += 450
    elif consultant_deal.invoice_frequency == InvoiceFrequency.SentBiWeekly:
        points += 650
    elif consultant_deal.invoice_frequency == InvoiceFrequency.SentMonthly:
        points += 850
    elif consultant_deal.invoice_frequency == InvoiceFrequency.SentSixWeeks:
        points += 1050
    elif consultant_deal.invoice_frequency == InvoiceFrequency.OnceAtEndOfSummer:
        points += 1250
    
    return points

def parse_deal_hfc(conversation_history, model, temperature):
    # This would need to be implemented based on the HFC negotiation format
    # For now, returning a placeholder
    return None

def parse_svi(message):
    # message = message.replace('Buyer:', '').replace('Seller:', '').strip()
    # This function would parse SVI responses
    # For now, returning a placeholder
    return None

def evaluate_negotiation(conversation, exercise, model="gpt-4o-mini", temperature=0.2):
    # This function would evaluate negotiation outcomes
    # For now, returning a placeholder
    return None 