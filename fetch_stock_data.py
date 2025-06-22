import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
from datetime import datetime

# Initialize API client
client = ApiClient()

# Define parameters for API call
symbol = "AAPL"
interval = "1d"
range_ = "10y"
include_adjusted_close = True

# Call the Yahoo Finance API
try:
    stock_data = client.call_api(
        'YahooFinance/get_stock_chart',
        query={
            'symbol': symbol,
            'interval': interval,
            'range': range_,
            'includeAdjustedClose': include_adjusted_close
        }
    )

    # Check if data was retrieved successfully
    if stock_data and 'chart' in stock_data and 'result' in stock_data['chart'] and stock_data['chart']['result']:
        result = stock_data['chart']['result'][0]
        timestamps = result.get('timestamp', [])
        indicators = result.get('indicators', {})
        quote = indicators.get('quote', [{}])[0]
        adjclose = indicators.get('adjclose', [{}])[0].get('adjclose', [])

        if timestamps and quote and adjclose and len(timestamps) == len(quote.get('open', [])):
            # Convert timestamps to datetime objects
            dates = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]

            # Create DataFrame
            df = pd.DataFrame({
                'Date': dates,
                'Open': quote.get('open', []), 
                'High': quote.get('high', []), 
                'Low': quote.get('low', []), 
                'Close': quote.get('close', []), 
                'Volume': quote.get('volume', []), 
                'Adj Close': adjclose
            })

            # Remove rows where any price data might be null (often happens at the start)
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj Close'], inplace=True)
            
            # Set Date as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Save to CSV
            output_file = "/home/ubuntu/aapl_stock_data_10y.csv"
            df.to_csv(output_file)
            print(f"Successfully fetched and saved data to {output_file}")
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        else:
            print("Error: Incomplete data received from API.")
            print(f"Timestamps: {len(timestamps)}, Opens: {len(quote.get('open', []))}, AdjClose: {len(adjclose)}")

    else:
        print("Error: Could not retrieve valid stock data from API.")
        print(f"API Response: {stock_data}")

except Exception as e:
    print(f"An error occurred: {e}")


