from django.urls import path, include
from .views import (home,stock_search, 
                    stock, balance_sheet, add_to_favorite_stocks, 
                    remove_from_favorite_stocks, portfolio, 
                    buy_stock, sell_stock)

urlpatterns = [
    path('', home, name='stock-home'),
    path('portfolio/', portfolio, name='portfolio'),
    path('stock_search/', stock_search, name='stock_search'),
    path('stock/<str:symbol>/', stock, name='stock'),
    path('balance_sheet/<str:stock_code>/', balance_sheet, name='balance_sheet'),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
    path('add_to_favorite_stocks/<str:symbol>/', add_to_favorite_stocks, name='add_to_favorite_stocks'),
    path('remove_from_favorite_stocks/<str:symbol>/', remove_from_favorite_stocks, name='remove_from_favorite_stocks'),
    path('buy_stock/<str:symbol>/', buy_stock, name='buy_stock'),
    path('sell_stock/<str:symbol>/', sell_stock, name='sell_stock'),

]
