# Generated by Django 2.2.5 on 2019-10-27 16:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0005_auto_20191027_1659'),
    ]

    operations = [
        migrations.AlterField(
            model_name='subject',
            name='gender',
            field=models.CharField(choices=[('m', 'male'), ('f', 'female')], default='f', max_length=200),
        ),
    ]