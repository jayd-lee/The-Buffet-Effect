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
    amount_invested = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    current_value = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    bought_at = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return self.symbol

class Portfolio(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    stocks = models.ManyToManyField(Stock)

    def __str__(self):
        return f'{self.user.username} Portfolio'