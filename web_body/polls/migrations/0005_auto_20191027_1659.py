# Generated by Django 2.2.5 on 2019-10-27 15:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0004_auto_20191027_1641'),
    ]

    operations = [
        migrations.AlterField(
            model_name='subject',
            name='gender',
            field=models.CharField(choices=[('m', 'male'), ('f', 'female')], max_length=200),
        ),
        migrations.AlterField(
            model_name='subject',
            name='img_f',
            field=models.ImageField(upload_to='images'),
        ),
        migrations.AlterField(
            model_name='subject',
            name='img_s',
            field=models.ImageField(upload_to='images'),
        ),
        migrations.AlterField(
            model_name='subject',
            name='measure',
            field=models.CharField(blank=True, max_length=10000),
        ),
    ]
