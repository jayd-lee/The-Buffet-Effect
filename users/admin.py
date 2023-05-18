from django.contrib import admin
from .models import Profile, FavoriteStocks, Portfolio, Stock

admin.site.register(Profile)
admin.site.register(FavoriteStocks)
admin.site.register(Portfolio)
admin.site.register(Stock)