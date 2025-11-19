import pandas as pd
import sys

def preprocess_bank_statement(input_file, output_file='transformed_transactions.csv'):
    """
    Transform bank statement CSV to required format for transaction categorization.
    
    Input format (your bank CSV):
    - Transaction_Date (DD-MM-YYYY HH:MM)
    - Description
    - Debit
    - Credit
    - Balance
    - Transaction_Mode
    - DR/CR_Indicator
    
    Output format (required):
    - date (YYYY-MM-DD HH:MM:SS)
    - amount (numeric)
    - description (string)
    - type (debit/credit)
    """
    
    print(f"📂 Reading {input_file}...")
    
    try:
        # Read the bank statement CSV
        df = pd.read_csv(input_file)
        
        print(f"✅ Loaded {len(df)} transactions")
        print(f"\n📋 Original columns: {list(df.columns)}")
        
        # Check if required columns exist
        required_input_cols = ['Transaction_Date', 'Description', 'Debit', 'Credit', 'DR/CR_Indicator']
        missing = [col for col in required_input_cols if col not in df.columns]
        
        if missing:
            print(f"❌ ERROR: Missing required columns: {missing}")
            return False
        
        # Step 1: Combine Debit and Credit into single 'amount' column
        print("\n🔧 Step 1: Combining Debit/Credit into amount...")
        df['amount'] = df['Debit'].fillna(0) + df['Credit'].fillna(0)
        
        # Remove rows with zero amount (if any)
        df = df[df['amount'] > 0].copy()
        print(f"   ✓ Combined amounts, {len(df)} valid transactions")
        
        # Step 2: Convert Transaction_Date to required format
        print("\n🔧 Step 2: Converting date format...")
        try:
            # Try DD-MM-YYYY HH:MM format first
            df['date'] = pd.to_datetime(df['Transaction_Date'], format='%d-%m-%Y %H:%M')
        except:
            try:
                # Try other common formats
                df['date'] = pd.to_datetime(df['Transaction_Date'])
            except Exception as e:
                print(f"   ❌ ERROR: Could not parse dates. Error: {e}")
                print(f"   Sample date value: {df['Transaction_Date'].iloc[0]}")
                return False
        
        # Convert to required string format
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        print(f"   ✓ Converted dates to YYYY-MM-DD HH:MM:SS format")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Step 3: Map DR/CR_Indicator to 'type'
        print("\n🔧 Step 3: Mapping transaction types...")
        df['type'] = df['DR/CR_Indicator'].str.strip().str.upper().map({
            'DR': 'debit',
            'CR': 'credit'
        })
        
        # Check for any unmapped types
        if df['type'].isna().any():
            print(f"   ⚠️  WARNING: Found {df['type'].isna().sum()} rows with unknown type")
            print(f"   Unique values in DR/CR_Indicator: {df['DR/CR_Indicator'].unique()}")
        
        type_counts = df['type'].value_counts()
        print(f"   ✓ Mapped transaction types:")
        print(f"     - Debit: {type_counts.get('debit', 0)} transactions")
        print(f"     - Credit: {type_counts.get('credit', 0)} transactions")
        
        # Step 4: Clean and use Description
        print("\n🔧 Step 4: Processing descriptions...")
        df['description'] = df['Description'].astype(str).str.strip()
        
        # Remove completely empty descriptions
        empty_desc = df['description'].str.len() == 0
        if empty_desc.any():
            print(f"   ⚠️  WARNING: Found {empty_desc.sum()} empty descriptions, removing...")
            df = df[~empty_desc].copy()
        
        print(f"   ✓ Processed {len(df)} descriptions")
        
        # Step 5: Sort by date chronologically (IMPORTANT!)
        print("\n🔧 Step 5: Sorting transactions chronologically...")
        df = df.sort_values('date').reset_index(drop=True)
        print(f"   ✓ Sorted {len(df)} transactions by date (oldest first)")
        
        # Step 6: Select only required columns
        print("\n🔧 Step 6: Selecting required columns...")
        df_final = df[['date', 'amount', 'description', 'type']].copy()
        
        # Optional: Add merchant column (copy from description initially)
        df_final['merchant'] = df_final['description']
        print(f"   ✓ Selected columns: {list(df_final.columns)}")
        
        # Step 7: Validate output
        print("\n🔍 Validating output...")
        
        # Check for missing values
        missing_check = df_final[['date', 'amount', 'description', 'type']].isna().sum()
        if missing_check.any():
            print(f"   ⚠️  WARNING: Found missing values:")
            print(missing_check[missing_check > 0])
        else:
            print(f"   ✓ No missing values in required columns")
        
        # Check amount range
        print(f"   ✓ Amount range: ₹{df_final['amount'].min():.2f} to ₹{df_final['amount'].max():.2f}")
        print(f"   ✓ Total transactions: {len(df_final)}")
        
        # Step 8: Save transformed CSV
        print(f"\n💾 Saving to {output_file}...")
        df_final.to_csv(output_file, index=False)
        print(f"   ✅ Successfully saved!")
        
        # Display preview
        print("\n" + "="*80)
        print("📊 PREVIEW OF TRANSFORMED DATA (first 10 rows):")
        print("="*80)
        print(df_final.head(10).to_string(index=False))
        print("="*80)
        
        # Display statistics
        print("\n📈 TRANSACTION STATISTICS:")
        print(f"   Total Transactions: {len(df_final)}")
        print(f"   Date Range: {df_final['date'].min()} to {df_final['date'].max()}")
        print(f"   Total Debit: ₹{df_final[df_final['type']=='debit']['amount'].sum():.2f}")
        print(f"   Total Credit: ₹{df_final[df_final['type']=='credit']['amount'].sum():.2f}")
        print(f"   Average Transaction: ₹{df_final['amount'].mean():.2f}")
        
        print("\n" + "="*80)
        print("✅ TRANSFORMATION COMPLETE!")
        print("="*80)
        print(f"\n📤 Next steps:")
        print(f"   1. Run: streamlit run app.py")
        print(f"   2. Open browser at http://localhost:8501")
        print(f"   3. Go to 'Upload & Classify' tab")
        print(f"   4. Upload: {output_file}")
        print(f"   5. Click '🚀 Start Classification'")
        print("="*80)
        
        return True
        
    except FileNotFoundError:
        print(f"❌ ERROR: File '{input_file}' not found!")
        print(f"   Please make sure the file exists in the current directory.")
        return False
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Default input file name
    input_file = 'bank_statement.csv'
    
    # Check if user provided a different filename
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    print("="*80)
    print("🏦 BANK STATEMENT PREPROCESSOR")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Output file: transformed_transactions.csv")
    print("="*80 + "\n")
    
    # Run preprocessing
    success = preprocess_bank_statement(input_file)
    
    if success:
        print("\n✅ All done! Your data is ready for classification.")
    else:
        print("\n❌ Preprocessing failed. Please check the errors above.")
        sys.exit(1)


