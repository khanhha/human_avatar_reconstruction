# Generated by Django 2.2.5 on 2019-10-28 10:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0009_auto_20191027_1753'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='subject',
            name='measure',
        ),
        migrations.AddField(
            model_name='subject',
            name='measure_gt',
            field=models.CharField(default='{"m_circ_bust": -1, "m_circ_underbust": -1, "m_circ_upperbust": -1, "m_circ_waist": -1, "m_circ_highhip": -1, "m_circ_hip": -1, "m_circ_thigh": -1, "m_circ_neck": -1}', max_length=10000),
        ),
        migrations.AddField(
            model_name='subject',
            name='measure_pred',
            field=models.CharField(default='{"m_circ_bust": -1, "m_circ_underbust": -1, "m_circ_upperbust": -1, "m_circ_waist": -1, "m_circ_highhip": -1, "m_circ_hip": -1, "m_circ_thigh": -1, "m_circ_neck": -1}', max_length=10000),
        ),
    ]
