# Generated by Django 4.2.1 on 2023-05-24 01:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0007_alter_stock_current_value_alter_stock_quantity"),
    ]

    operations = [
        migrations.AlterField(
            model_name="stock",
            name="amount_invested",
            field=models.DecimalField(decimal_places=2, default=0, max_digits=100),
        ),
        migrations.AlterField(
            model_name="stock",
            name="bought_at",
            field=models.DecimalField(decimal_places=2, max_digits=100),
        ),
        migrations.AlterField(
            model_name="stock",
            name="current_price",
            field=models.DecimalField(decimal_places=2, default=0, max_digits=100),
        ),
        migrations.AlterField(
            model_name="stock",
            name="current_value",
            field=models.DecimalField(decimal_places=8, default=0, max_digits=100),
        ),
        migrations.AlterField(
            model_name="stock",
            name="quantity",
            field=models.DecimalField(decimal_places=8, default=0, max_digits=100),
        ),
    ]
