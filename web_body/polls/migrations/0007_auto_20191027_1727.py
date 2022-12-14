# Generated by Django 2.2.5 on 2019-10-27 16:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0006_auto_20191027_1701'),
    ]

    operations = [
        migrations.AlterField(
            model_name='subject',
            name='gender',
            field=models.CharField(choices=[('male', 'm'), ('female', 'f')], default='female', max_length=200),
        ),
        migrations.AlterField(
            model_name='subject',
            name='img_f',
            field=models.ImageField(upload_to='polls/images'),
        ),
        migrations.AlterField(
            model_name='subject',
            name='img_result',
            field=models.ImageField(blank=True, upload_to='polls/images'),
        ),
        migrations.AlterField(
            model_name='subject',
            name='img_s',
            field=models.ImageField(upload_to='polls/images'),
        ),
    ]
