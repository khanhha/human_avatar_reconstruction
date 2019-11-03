import datetime

from django.db import models
from django.utils import timezone
from django.core.files.storage import FileSystemStorage
import json

mids = ['m_breast_depth', 'm_cross_breast', 'm_circ_bust',
        'm_circ_waist', 'm_circ_hip', 'm_len_front_body',
        'm_len_half_girth', 'm_len_bikini_girth', 'm_len_full_girth', 'm_circ_neck',
        'm_circ_upperbust', 'm_circ_underbust', 'm_circ_highhip',
        'm_circ_thigh', 'm_circ_knee', 'm_circ_upperarm', 'm_circ_elbow',
        'm_circ_wrist', 'm_shoulder_to_bust', 'm_len_upperarm',
        'm_len_sleeve', 'm_len_waist_knee', 'm_len_skirt_waist_to_hem']

def _default_measure_json():
    measures = dict((id, -1) for id in mids)
    return _measure_to_json(measures)

def _json_to_measure(json_str):
    return json.loads(json_str)

def _measure_to_json(measures):
    return json.dumps(measures)

class Subject(models.Model):
    name = models.CharField(max_length=200)
    gender = models.CharField(max_length=200, choices=[('male','m'), ('female', 'f')], default='female')
    height = models.FloatField(default=1.5)
    measure_gt = models.CharField(max_length=10000, default = _default_measure_json())
    measure_pred = models.CharField(max_length=10000, default = _default_measure_json())
    img_f  = models.ImageField(upload_to='images')
    img_s =  models.ImageField(upload_to='images')

    img_sil_f =  models.ImageField(upload_to='body_result_sil', blank=True)
    img_sil_s =  models.ImageField(upload_to='body_result_sil', blank=True)

    img_result          = models.ImageField(upload_to='body_result_viz', blank=True)
    img_measure_viz  = models.ImageField(upload_to='body_result_viz', blank=True)

    mesh_path =  models.FilePathField(match = "*.obj", blank=True)

    # def __str__(self):
    #     return self.name

    @property
    def measures_gt(self):
        return _json_to_measure(self.measure_gt)

    @measures_gt.setter
    def measures_gt(self, measures):
        self.measure_gt = _measure_to_json(measures)

    @property
    def measures_pred(self):
        return _json_to_measure(self.measure_pred)

    @measures_pred.setter
    def measures_pred(self, measures):
        self.measure_pred = _measure_to_json(measures)

class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.question_text

    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now
    was_published_recently.admin_order_field = 'pub_date'
    was_published_recently.boolean = True
    was_published_recently.short_description = 'Published recently?'


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text
