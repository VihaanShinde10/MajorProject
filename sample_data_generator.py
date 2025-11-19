import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_transactions(n=200):
    """Generate sample transaction data for testing."""
    
    categories = {
        'Food & Dining': {
            'merchants': ['Swiggy', 'Zomato', 'McDonald', 'Starbucks', 'LocalRestaurant'],
            'amounts': (50, 800),
            'type': 'debit'
        },
        'Commute/Transport': {
            'merchants': ['Uber', 'Ola', 'Metro', 'RapidoUPI@okaxis', 'PetrolPump'],
            'amounts': (30, 500),
            'type': 'debit'
        },
        'Shopping': {
            'merchants': ['Amazon', 'Flipkart', 'BigBazaar', 'Mall', 'OnlineStore'],
            'amounts': (200, 5000),
            'type': 'debit'
        },
        'Bills & Utilities': {
            'merchants': ['ElectricityBill', 'WaterBill', 'BroadbandBill', 'GasBill'],
            'amounts': (500, 2000),
            'type': 'debit'
        },
        'Subscriptions': {
            'merchants': ['Netflix', 'Spotify', 'AmazonPrime', 'Gym'],
            'amounts': (199, 999),
            'type': 'debit'
        },
        'Investments': {
            'merchants': ['MutualFundSIP', 'Zerodha', 'StockInvestment'],
            'amounts': (1000, 5000),
            'type': 'debit'
        },
        'Salary/Income': {
            'merchants': ['SalaryCredit', 'CompanyName', 'PayrollDeposit'],
            'amounts': (30000, 80000),
            'type': 'credit'
        }
    }
    
    transactions = []
    start_date = datetime.now() - timedelta(days=180)
    
    for i in range(n):
        # Random date
        days_ago = random.randint(0, 180)
        txn_date = start_date + timedelta(days=days_ago)
        
        # Random time
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        txn_date = txn_date.replace(hour=hour, minute=minute)
        
        # Random category
        category = random.choice(list(categories.keys()))
        cat_data = categories[category]
        
        # Generate transaction
        merchant = random.choice(cat_data['merchants'])
        amount = round(random.uniform(*cat_data['amounts']), 2)
        txn_type = cat_data['type']
        
        # Add some noise to merchant names
        if random.random() > 0.7:
            merchant = f"UPI-{merchant}-REF{random.randint(100000, 999999)}"
        
        description = f"{merchant} Payment"
        if random.random() > 0.5:
            description += f" #{random.randint(1000, 9999)}"
        
        # Determine transaction mode (UPI, INB, etc.)
        modes = ['UPI', 'INB', 'CARD', 'NEFT', 'IMPS']
        mode = random.choice(modes)
        
        # Create debit/credit columns
        debit = amount if txn_type == 'debit' else 0
        credit = amount if txn_type == 'credit' else 0
        
        transactions.append({
            'Transaction_Date': txn_date.strftime('%d-%m-%Y %H:%M'),
            'Description': description,
            'Debit': debit,
            'Credit': credit,
            'Balance': 0,  # Placeholder, will be calculated
            'Transaction_Mode': mode,
            'DR/CR_Indicator': 'DR' if txn_type == 'debit' else 'CR',
            'merchant': merchant,  # Extra field for internal use
            'true_category': category  # For validation
        })
    
    df = pd.DataFrame(transactions)
    df = df.sort_values('Transaction_Date').reset_index(drop=True)
    
    # Calculate running balance (starting with 50,000)
    balance = 50000
    balances = []
    for idx, row in df.iterrows():
        if row['DR/CR_Indicator'] == 'DR':
            balance -= row['Debit']
        else:
            balance += row['Credit']
        balances.append(round(balance, 2))
    df['Balance'] = balances
    
    return df

if __name__ == '__main__':
    df = generate_sample_transactions(200)
    df.to_csv('sample_transactions.csv', index=False)
    print(f"✅ Generated {len(df)} sample transactions")
    print(f"\nCategory distribution:")
    print(df['true_category'].value_counts())
    print(f"\nSaved to sample_transactions.csv")

