import argparse
from binanceClassErrors import *
from binance_class import BinanceOption
import sys
from datetime import datetime
import struct

YEARLY_MILLI_SECONDS = 365*24*3600*1000


def get_strike_price(optionStr: str):
    """Extracts strike from contract name in ex: 'BTC-240229-58000-C'"""
    splitted_option = optionStr.split('-')
    
    return float(splitted_option[2])

def main():
    parser = argparse.ArgumentParser(description="Takes in coin symbol (BTC/ETH/DOGE), side (BID/ASK) and option type (P/C). \
                                     Writes to a binary file: symbol_Side_Date_OptionType.bin in format: \
                                     maturity(years)  strike  (BID/ASK)price ")
    
    parser.add_argument("--symbol", type=str, help="Coin symbol to process")
    parser.add_argument("--side", type=str, help="Get price from bid or ask side")
    parser.add_argument("--type", type=str, help="Call or put option")

    args = parser.parse_args()
    
    if len(sys.argv) != 4:
        raise ValueError(f"Error: Exptected 3 arguments: symbol, side and option type\n.")
    
    symbol = args.symbol.lower()
    side = args.side.lower()
    optionType = args.type.upper()

    if symbol not in ['btc', 'eth', 'doge']:
        raise symbolNotFoundError(f"Error: Symbol '{symbol}' not permitted. Input one of btc/eth/doge\n")
    if side not in ['bid', 'ask']:
        raise sideNotFoundError(f"Error: Side '{side}' not permitted. Should be ask/bid\n")
    if optionType not in ('P', 'C'):
        raise optionTypeNotFoundError(f"Error: Optiontype '{optionType}' not permitted. Should be P/C\n")
       
    
    btc = BinanceOption(symbol)
    btc.init_option_data()
    btc.get_underlying_coin_price()

    keys = btc.get_optionData_keys()
    
    todaysdate = datetime.today()
    formatted_date = todaysdate.strftime("%y%m%d")

    filename = symbol + '_' + side.upper() + '_' + formatted_date + '_' + optionType + ".bin"
    row = [0]*3

    with open(filename, 'wb') as f:
        for option in btc.option_data:

            side_price = option[side +"Price"]
            time_to_maturity = (option['expirationTime'] - btc.unix_epoch_time) / YEARLY_MILLI_SECONDS
            strike = get_strike_price(option['symbol'])
            vol = option['volatility']
            if optionType  == option['symbol'][-1] and float(side_price) != 0.0:
               
                
                row[0] = time_to_maturity
                row[1] = strike
                row[2] = side_price
          
                 
                for data in row:
                    packed_data = struct.pack('d', float(data))
                    f.write(packed_data)
            elif optionType  == option['symbol'][-1] and float(side_price) == 0.0:
                print(f"Option with symbol {option['symbol']} has price {side_price}\n")
    print(f"Wrote to {filename}\n")
    
if __name__ == "__main__":
    main()
