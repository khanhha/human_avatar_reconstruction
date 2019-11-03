from django.urls import path
from . import views

app_name = 'polls'
urlpatterns = [
    path('', views.subject_list, name='subject-list'),
    path('add', views.add_subject, name='add-subject'),
    path('detail/<int:subject_id>/', views.subject_detail, name='subject-detail'),
    path('update_mgt/<int:subject_id>/', views.update_mgt, name='update-mgt'),
    path('predict_shape/<int:subject_id>/', views.predict_shape, name='predict-shape'),
    path('predict_measure/<int:subject_id>/', views.predict_measure, name='predict-measure'),
    path('delete_subject/<int:subject_id>/', views.delete_subject, name='delete-subject'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    path('<int:question_id>/vote/', views.vote, name='vote'),
]
