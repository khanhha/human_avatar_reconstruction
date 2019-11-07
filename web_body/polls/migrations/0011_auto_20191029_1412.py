# Generated by Django 2.2.5 on 2019-10-29 13:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0010_auto_20191028_1137'),
    ]

    operations = [
        migrations.AddField(
            model_name='subject',
            name='img_sil_f',
            field=models.ImageField(blank=True, upload_to='body_result_images'),
        ),
        migrations.AddField(
            model_name='subject',
            name='img_sil_s',
            field=models.ImageField(blank=True, upload_to='body_result_images'),
        ),
        migrations.AddField(
            model_name='subject',
            name='mesh_path',
            field=models.FilePathField(blank=True, match='*.obj'),
        ),
        migrations.AlterField(
            model_name='subject',
            name='img_result',
            field=models.ImageField(blank=True, upload_to='body_result_images'),
        ),
    ]