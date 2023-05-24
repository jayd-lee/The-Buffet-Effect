from django.db import models
from django.contrib.auth.models import User
from PIL import Image


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')

    def __str__(self):
        return f'{self.user.username} Profile'

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        img = Image.open(self.image.path)

        if img.height > 300 or img.width > 300:
            output_size = (300, 300)
            img.thumbnail(output_size)
            img.save(self.image.path)


class FavoriteStocks(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    symbol = models.CharField(max_length=10)

    def __str__(self):
        return f'{self.user.username} - {self.symbol}'
    
class Stock(models.Model):
    symbol = models.CharField(max_length=10)
    amount_invested = models.DecimalField(max_digits=100, decimal_places=2, default=0)
    current_value = models.DecimalField(max_digits=100, decimal_places=8, default=0)
    current_price = models.DecimalField(max_digits=100, decimal_places=2, default=0)
    bought_at = models.DecimalField(max_digits=100, decimal_places=2)
    quantity = models.DecimalField(max_digits=100, decimal_places=8, default=0)

    def __str__(self):
        return self.symbol
    
    def update_current_value(self):
        self.current_value = self.quantity * self.current_price
        self.save()
    
    @property
    def gain_loss(self):
        return (self.current_value - self.amount_invested)  # use amount_invested instead of bought_at

    @property
    def gain_loss_pct(self):
        return (self.gain_loss * 100 / self.amount_invested) if self.amount_invested else 0  # use gain_loss and amount_invested

class Portfolio(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    stocks = models.ManyToManyField(Stock)

    def __str__(self):
        return f'{self.user.username} Portfolio'
    
    @property
    def total_invested(self):
        return sum(stock.amount_invested for stock in self.stocks.all())

    @property
    def total_value(self):
        return sum(stock.current_value for stock in self.stocks.all())

    @property
    def avg_gain_loss_pct(self):
        total_gain_loss_pct = sum(stock.gain_loss_pct for stock in self.stocks.all())
        num_stocks = self.stocks.count()
        return total_gain_loss_pct / num_stocks if num_stocks > 0 else 0

    @property
    def status(self):
        return "Gain" if self.total_value > self.total_invested else "Loss" if self.total_value < self.total_invested else "-"