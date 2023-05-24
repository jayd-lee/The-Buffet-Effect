from django.shortcuts import render, redirect, HttpResponse
import pandas as pd
import requests
import numpy as np
import plotly.graph_objs as go
from django_plotly_dash import DjangoDash
from dash import html, dcc
from users.models import FavoriteStocks, Portfolio, Stock
import json
from django.contrib.auth.decorators import login_required
from decimal import Decimal


def home(request):
    if request.user.is_authenticated:
        watchlist = FavoriteStocks.objects.filter(user=request.user).values_list('symbol', flat=True)
    else:
        watchlist = []

    watchlist = list(map(lambda x: [x], watchlist))
    widget_config = {
        "container_id": "tradingview-widget-container",
        "symbols": watchlist,
        "chartOnly": False,
        "width": "100%",
        "height": "100%",
        "locale": "en",
        "colorTheme": "light",
        "autosize": True,
        "showVolume": False,
        "showMA": False,
        "hideDateRanges": False,
        "hideMarketStatus": False,
        "hideSymbolLogo": False,
        "scalePosition": "right",
        "scaleMode": "Normal",
        "fontFamily": "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
        "fontSize": "10",
        "noTimeScale": False,
        "valuesTracking": "1",
        "changeMode": "price-and-percent",
        "chartType": "area",
        "lineWidth": 2,
        "lineType": 0,
        "dateRanges": [
            "1d|1",
            "1m|30",
            "3m|60",
            "12m|1D",
            "60m|1W",
            "all|1M"
        ]
    }
    config_str = json.dumps(widget_config).replace("</", "<\\/")  # Escape closing tags to prevent breaking out of the script tag

    widget_script = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js" async>
      {config_str}
      </script>
    </div>
    """

    context = {
        'widget_script': widget_script
    }

    return render(request, 'stock/home.html', context)


def about(request):
    return render(request, 'stock/about.html', {'title': 'About'})




header = {'User-Agent': "mr.muffin235@gmail.com"}
companyTickers = requests.get("https://www.sec.gov/files/company_tickers.json", headers=header)
T_list = companyTickers.json().values()
#symbol = input('Enter company ticker symbol: ').upper()


pd.set_option('display.max_columns', None)


import alpaca_trade_api as tradeapi
from alpaca.data import TimeFrame
from .key import *




def balance_sheet(request, stock_code):

    cik = ""

    for info in T_list:
        if stock_code == info['ticker']:
            new = info['cik_str'] #if there is a match assign to a new variable
            cik = f'{new:010d}' #format the new variable with leading zeros
    
    companyConcept = requests.get(
        (f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'), headers=header)

    pd.options.display.float_format = '{:.0f}'.format

    def show(account):
        df_list = companyConcept.json()['facts']['us-gaap'][account]
        df = pd.DataFrame(df_list['units']['USD'])
        df_final = df[(df['fp'] == 'FY') & (df['form'] == '10-K')] #gets only the rows with 'form' == 10-K, and etc
        df_final = df_final.drop_duplicates('end') #drops duplicate 'end'
        df_final = df_final.drop_duplicates('val') #drops duplicate 'val'
        
        df_final = df_final.drop(columns=['accn', 'filed', 'fp', 'fy', 'frame', 'form',]) #drop columns that arent needed
        df_final = df_final.set_index('end') #sets index to fy
        df_final.columns = [account]

        return df_final
        
    try:
        assets = show('Assets')
    except:
        assets = pd.DataFrame(np.nan, index=['fy'], columns=['assets'])

    try:
        stock_holder_equity = show('StockholdersEquity')
    except:
        try:
            stock_holder_equity = show('StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')
            stock_holder_equity = stock_holder_equity.rename(columns={'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest': 'Equity2'})
        except:
            stock_holder_equity = pd.DataFrame(np.nan, index=['fy'], columns=['stockholdersequity'])

    result = pd.merge(assets, stock_holder_equity, left_index=True, right_index=True, how='outer')

    try:
        liabilities = show('Liabilities')
        result = pd.merge(result, liabilities, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        result.insert(1, 'Liabilities', result.iloc[:, 0] - result.iloc[:, 1])
        result = result.sort_index(ascending=False)

    try:
        assets_current = show('AssetsCurrent')
        result = pd.merge(result, assets_current, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        assets_current = pd.DataFrame(np.nan, index=['fy'], columns=['AssetsCurrent'])

    try:
        liabilities_current = show('LiabilitiesCurrent')
        result = pd.merge(result, liabilities_current, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        liabilities_current = pd.DataFrame(np.nan, index=['fy'], columns=['LiabilitiesCurrent'])

    try:
        cash = show('CashAndCashEquivalentsAtCarryingValue')
        result = pd.merge(result, cash, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        pd.DataFrame(np.nan, index=['fy'], columns=['CashAndCashEquivalentsAtCarryingValue'])

    try:
        stock_value = show('CommonStockValue')
        result = pd.merge(result, stock_value, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        stock_value = pd.DataFrame(np.nan, index=['fy'], columns=['CommonStockValue'])

    try:
        com_stock_par = show('CommonStockParOrStatedValuePerShare')
        result = pd.merge(result, com_stock_par, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        com_stock_par = pd.DataFrame(np.nan, index=['fy'], columns=['CommonStockParOrStatedValuePerShare'])

    try:
        ppe = show('PropertyPlantAndEquipmentNet')
        result = pd.merge(result, ppe, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        ppe = pd.DataFrame(np.nan, index=['fy'], columns=['PropertyPlantAndEquipmentNet'])

    try:
        ppe_depr = show('AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment')
        result = pd.merge(result, ppe_depr, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        ppe_depr = pd.DataFrame(np.nan, index=['fy'], columns=['AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment'])

    try:
        assets_leases = show('OperatingLeaseRightOfUseAsset')
        result = pd.merge(result, assets_leases, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        assets_leases = pd.DataFrame(np.nan, index=['fy'], columns=['OperatingLeaseRightOfUseAsset'])

    try:
        assets_current = show('DeferredTaxAssetsGross')
        result = pd.merge(result, assets_current, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        assets_current = pd.DataFrame(np.nan, index=['fy'], columns=['DeferredTaxAssetsGross'])

    try:
        apic = show('AdditionalPaidInCapital')
        result = pd.merge(result, apic, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        apic = pd.DataFrame(np.nan, index=['fy'], columns=['AdditionalPaidInCapital'])

    try:
        liabilities_leases = show('OperatingLeaseLiability')
        result = pd.merge(result, liabilities_leases, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        liabilities_leases = pd.DataFrame(np.nan, index=['fy'], columns=['OperatingLeaseLiability'])

    try:
        proceeds = show('ProceedsFromIssuanceOfCommonStock')
        result = pd.merge(result, proceeds, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        proceeds = pd.DataFrame(np.nan, index=['fy'], columns=['ProceedsFromIssuanceOfCommonStock'])

    try:
        prf_stock_par = show('PreferredStockParOrStatedValuePerShare')
        result = pd.merge(result, prf_stock_par, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        prf_stock_par = pd.DataFrame(np.nan, index=['fy'], columns=['PreferredStockParOrStatedValuePerShare'])

    try:
        ppe_gross = show('PropertyPlantAndEquipmentGross')
        result = pd.merge(result, ppe_gross, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        ppe_gross = pd.DataFrame(np.nan, index=['fy'], columns=['PropertyPlantAndEquipmentGross'])

    try:
        assets_tax = show('DeferredTaxAssetsNet')
        result = pd.merge(result, assets_tax, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        assets_tax = pd.DataFrame(np.nan, index=['fy'], columns=['DeferredTaxAssetsNet'])

    try:
        ap_current = show('AccountsPayableCurrent')
        result = pd.merge(result, ap_current, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        ap_current = pd.DataFrame(np.nan, index=['fy'], columns=['AccountsPayableCurrent'])

    try:
        liabilities_leases_current = show('OperatingLeaseLiabilityCurrent')
        result = pd.merge(result, liabilities_leases_current, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        liabilities_leases_current = pd.DataFrame(np.nan, index=['fy'], columns=['OperatingLeaseLiabilityCurrent'])

    try:
        liabilities_leases_noncurrent = show('OperatingLeaseLiabilityNoncurrent')
        result = pd.merge(result, liabilities_leases_noncurrent, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        liabilities_leases_noncurrent = pd.DataFrame(np.nan, index=['fy'], columns=['OperatingLeaseLiabilityNoncurrent'])

    try:
        liabilities_tax = show('DeferredTaxAssetsLiabilitiesNet')
        result = pd.merge(result, liabilities_tax, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        liabilities_tax = pd.DataFrame(np.nan, index=['fy'], columns=['DeferredTaxAssetsLiabilitiesNet'])

    try:
        ar_current = show('AccountsReceivableNetCurrent')
        result = pd.merge(result, ar_current, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        ar_current = pd.DataFrame(np.nan, index=['fy'], columns=['AccountsReceivableNetCurrent'])

    try:
        assets_other = show('OtherAssetsNoncurrent')
        result = pd.merge(result, assets_other, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        assets_other = pd.DataFrame(np.nan, index=['fy'], columns=['OtherAssetsNoncurrent'])

    try:
        deferred = show('DeferredTaxAssetsOther')
        result = pd.merge(result, deferred, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        deferred = pd.DataFrame(np.nan, index=['fy'], columns=['DeferredTaxAssetsOther'])

    try:
        accrued_liabilities_current = show('AccruedLiabilitiesCurrent')
        result = pd.merge(result, accrued_liabilities_current, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        accrued_liabilities_current = pd.DataFrame(np.nan, index=['fy'], columns=['AccruedLiabilitiesCurrent'])

    try:
        goodwill = show('Goodwill')
        result = pd.merge(result, goodwill, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        goodwill = pd.DataFrame(np.nan, index=['fy'], columns=['Goodwill'])

    try:
        deferred_tax = show('DeferredIncomeTaxLiabilities')
        result = pd.merge(result, deferred_tax, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        deferred_tax = pd.DataFrame(np.nan, index=['fy'], columns=['DeferredIncomeTaxLiabilities'])

    try:
        debt = show('LongTermDebt')
        result = pd.merge(result, debt, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        debt = pd.DataFrame(np.nan, index=['fy'], columns=['LongTermDebt'])

    try:
        intangible = show('FiniteLivedIntangibleAssetsNet')
        result = pd.merge(result, intangible, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        intangible = pd.DataFrame(np.nan, index=['fy'], columns=['FiniteLivedIntangibleAssetsNet'])

    try:
        intangible_depr = show('FiniteLivedIntangibleAssetsAccumulatedAmortization')
        result = pd.merge(result, intangible_depr, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        intangible_depr = pd.DataFrame(np.nan, index=['fy'], columns=['FiniteLivedIntangibleAssetsAccumulatedAmortization'])

    try:
        debt_current = show('LongTermDebtCurrent')
        result = pd.merge(result, debt_current, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        debt_current = pd.DataFrame(np.nan, index=['fy'], columns=['LongTermDebtCurrent'])

    try:
        retained_earnings = show('RetainedEarningsAccumulatedDeficit')
        result = pd.merge(result, retained_earnings, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False)
    except:
        retained_earnings = pd.DataFrame(np.nan, index=['fy'], columns=['RetainedEarningsAccumulatedDeficit'])

    try:
        liabilities_equity = show('LiabilitiesAndStockholdersEquity')
        result = pd.merge(result, liabilities_equity, left_index=True, right_index=True, how='outer')
        result = result.sort_index(ascending=False).T
    except:
        liabilities_equity = pd.DataFrame(np.nan, index=['fy'], columns=['LiabilitiesAndStockholdersEquity'])
    
    
    result = result.applymap('{:,.0f}'.format)

    context ={
        'result': result,
        'symbol': stock_code
    }
    
    return render(request, 'stock/balance_sheet.html', context)


def get_historical_stock(symbol, start_year=None, end_year=None):
    pd.set_option('display.float_format', '{:.2f}'.format)
    # create Alpaca API object with the given credentials
    api = tradeapi.REST(API_KEY,
                        SECRET_KEY,
                        base_url)
    status = True
    if start_year is None:
    # Call the API to get OHLC TSLA data adn store it in a dataframe
        data = api.get_bars(
            symbol=symbol,
            timeframe=TimeFrame.Minute
        ).df

    else:
        # Get some historical data for TSLA
        data = api.get_bars(
            symbol='TSLA', #any symbol is acceptable if it can be found in Alpaca API
            timeframe=TimeFrame.Minute,
            start="2018-01-01T00:00:00-00:00",
            end="2018-02-01T00:00:00-00:00"
        ).df
        status=False
    return data, status

def stock_search(request):
    if request.method == 'GET':
        query_dict = request.GET
        query = query_dict.get('q')
        return redirect('stock', symbol=query)

def stock(request, symbol=None):
    if symbol is None:
        if request.method == 'GET':
            query_dict = request.GET
            query = query_dict.get('q')
            if query:
                symbol = query.upper()
            else:
                # You might want to handle the case where no symbol is provided differently
                return HttpResponse("No stock symbol provided")

    # API calls
    historical_stock, status = get_historical_stock(symbol)
    historical_stock_json = historical_stock.reset_index().to_json(date_format='iso')
   

    if request.user.is_authenticated:
        watchlist = FavoriteStocks.objects.filter(user=request.user).values_list('symbol', flat=True)
    else:
        watchlist = []
    context = {
        'historical_stock': historical_stock,
        'historical_stock_json': historical_stock_json,
        'stock_code':symbol.upper(),
        'status':status,
        'watchlist': watchlist
        
        }
    
    return render(request, 'stock/stock.html', context=context)


def add_to_favorite_stocks(request, symbol):
    if request.user.is_authenticated:
        FavoriteStocks.objects.get_or_create(user=request.user, symbol=symbol)
    return redirect('stock', symbol=symbol)

def remove_from_favorite_stocks(request, symbol):
    if request.user.is_authenticated:
        FavoriteStocks.objects.filter(user=request.user, symbol=symbol).delete()
    return redirect('stock', symbol=symbol)


def buy_stock(request, symbol):
    if request.method == 'POST':
        api = tradeapi.REST(API_KEY, SECRET_KEY, base_url='https://paper-api.alpaca.markets')
        amount = Decimal(request.POST.get('amount')) 

        data = api.get_bars(
            symbol=symbol,
            timeframe=TimeFrame.Minute
        ).df

        # Get the most recent price
        current_price = Decimal(data.iloc[-1]['close']) 
        quantity = Decimal(amount / current_price)

        api.submit_order(
            symbol=symbol,
            qty=float(quantity),
            side='buy',
            type='market',
            time_in_force='day'
        )

        # First try to get the stock
        try:
            stock = request.user.portfolio.stocks.get(symbol=symbol)
        except Stock.DoesNotExist:
            # If it does not exist, create it
            stock = Stock.objects.create(symbol=symbol, amount_invested=amount, current_value=quantity*current_price, bought_at=current_price, quantity=quantity)
            request.user.portfolio.stocks.add(stock)
        else:
            # If it exists, update its fields
            stock.amount_invested += amount
            stock.current_value += quantity*current_price
            stock.quantity += quantity

            stock.current_value = stock.quantity * current_price

            stock.save()

        return redirect('portfolio')

    

def sell_stock(request, symbol):
    if request.method == 'POST':
        api = tradeapi.REST(API_KEY, SECRET_KEY, base_url)
        amount = Decimal(request.POST.get('amount')) 

        stock = request.user.portfolio.stocks.get(symbol=symbol)

        data = api.get_bars(
            symbol=symbol,
            timeframe=TimeFrame.Minute
        ).df

        # Get the most recent price
        current_price = Decimal(data.iloc[-1]['close']) 

        quantity = Decimal(amount / current_price)

        # Check if enough shares are available
        if quantity > stock.quantity:
            quantity = stock.quantity

        api.submit_order(
            symbol=symbol,
            qty=float(quantity),
            side='sell',
            type='market',
            time_in_force='day'
        )

        stock.amount_invested -= amount
        stock.current_value -= quantity*current_price
        stock.quantity -= quantity

        stock.current_value = stock.quantity * current_price

        if stock.quantity <= 0:
            request.user.portfolio.stocks.remove(stock)
        else:
            stock.save()

    return redirect('portfolio')

@login_required
def portfolio(request):
    api = tradeapi.REST(API_KEY, SECRET_KEY, base_url)
    portfolio = request.user.portfolio
    for stock in portfolio.stocks.all():
        data = api.get_bars(symbol=stock.symbol, timeframe=TimeFrame.Minute).df
        stock.current_price = Decimal(data.iloc[-1]['close'])  # Make sure this is Decimal, not float
        stock.update_current_value()  # Update current value whenever the current price changes
    return render(request, 'stock/portfolio.html', {'portfolio': portfolio})


