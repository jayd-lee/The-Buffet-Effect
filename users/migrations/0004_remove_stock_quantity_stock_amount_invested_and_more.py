# Generated by Django 4.2.1 on 2023-05-17 23:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0003_stock_portfolio"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="stock",
            name="quantity",
        ),
        migrations.AddField(
            model_name="stock",
            name="amount_invested",
            field=models.DecimalField(decimal_places=2, default=0, max_digits=10),
        ),
        migrations.AddField(
            model_name="stock",
            name="current_value",
            field=models.DecimalField(decimal_places=2, default=0, max_digits=10),
        ),
    ]
