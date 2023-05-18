from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import Profile, Portfolio


@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_profile(sender, instance, **kwargs):
    instance.profile.save()


@receiver(post_save, sender=User)
def create_portfolio(sender, instance, created, **kwargs):
    if created:
        Portfolio.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_portfolio(sender, instance, **kwargs):
    instance.portfolio.save()