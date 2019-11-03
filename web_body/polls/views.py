from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.shortcuts import redirect
from django.conf import settings
from .models import Choice, Question, Subject
from PIL import Image
from pathlib import Path
from skimage.io import imread
from io import BytesIO
from .inference import infer_mesh, infer_mesurement, build_shape_visualization
from common.obj_util import export_mesh, import_mesh_obj
from .forms import SubjectForm
from django.core.files import File
import numpy as np

def add_subject(request):
    if request.method == 'POST':
        form = SubjectForm(request.POST, request.FILES)
        if form.is_valid():
            sub = form.save()
            return redirect('polls:subject-detail', subject_id=sub.id)
    else:
        form = SubjectForm()

    return render(request, 'polls/add_subject.html', {
        'form': form
    })

def delete_subject(request, subject_id):
    sub = Subject.objects.get(id=subject_id)
    sub.delete()
    return redirect("polls:subject-list")

def subject_detail(request, subject_id):
    sub = Subject.objects.get(id=subject_id)
    return render(request, 'polls/subject_detail.html', {'subject':sub})

def predict_shape(request, subject_id):
    sub = Subject.objects.get(id=subject_id)
    media_dir = Path(settings.MEDIA_ROOT)
    img_f = imread(sub.img_f.path)
    img_s = imread(sub.img_s.path)
    gender_val = 0 if sub.gender in ['f', 'female'] else 1

    verts, triangles, sil_f, sil_s = infer_mesh(img_f, img_s, sub.height, gender_val)
    sil_f_rgb = np.dstack([sil_f, sil_f, sil_f])
    sil_s_rgb = np.dstack([sil_s, sil_s, sil_s])

    #save results
    mesh_path = media_dir/Path(f'body_result_mesh/{Path(sub.img_f.path).stem}_shape.obj')
    sub.mesh_path = str(mesh_path)
    export_mesh(mesh_path, verts, triangles)

    blob = BytesIO()
    sil_f_pil = Image.fromarray(sil_f_rgb)
    sil_f_pil.save(blob, 'JPEG')
    sub.img_sil_f.save(f'{Path(sub.img_f.path).stem}_sil_f.jpg', File(blob), save=False)

    blob = BytesIO()
    sil_s_pil = Image.fromarray(sil_s_rgb)
    sil_s_pil.save(blob, 'JPEG')
    sub.img_sil_s.save(f'{Path(sub.img_s.path).stem}_sil_s.jpg', File(blob), save=False)

    img_viz = build_shape_visualization(verts, sil_f_rgb, sil_s_rgb)
    img_viz_pil = Image.fromarray(img_viz)
    blob = BytesIO()
    img_viz_pil.save(blob, 'JPEG')
    sub.img_result.save(f'{Path(sub.img_f.path).stem}_viz.jpg', File(blob), save=False)

    sub.save()

    return redirect('polls:subject-detail', subject_id=sub.id)

def predict_measure(request, subject_id):
    sub = Subject.objects.get(id=subject_id)
    obj_path = Path(sub.mesh_path)
    if obj_path.exists() and Path(sub.img_sil_f.path).exists() and Path(sub.img_sil_s.path).exists():
        verts, triangles = import_mesh_obj(obj_path)

        new_measures, contours = infer_mesurement(verts, sub.height)

        #update measurement
        cur_measure = sub.measures_pred
        for key, value in new_measures.items():
            if key in cur_measure.keys():
                cur_measure[key] = value
        sub.measures_pred = cur_measure

        #build measure visualization
        img_sil_f = imread(sub.img_sil_f.path)
        img_sil_s = imread(sub.img_sil_s.path)
        img_measure_viz = build_shape_visualization(verts, img_sil_f, img_sil_s, measure_contours=contours, ortho_proj=True, body_opacity=0.5)

        #save visualization
        img_viz_pil = Image.fromarray(img_measure_viz)
        blob = BytesIO()
        img_viz_pil.save(blob, 'JPEG')
        sub.img_measure_viz.save(f'{Path(sub.img_f.path).stem}_viz.jpg', File(blob), save=False)

        sub.save()

    return redirect('polls:subject-detail', subject_id=sub.id)

#measure update
def update_mgt(request, subject_id):
    if request.method == 'POST':
        sub = Subject.objects.get(id = subject_id)
        if sub:
            cur_measures = sub.measures_gt
            for k in cur_measures.keys():
                if k in request.POST:
                    cur_measures[k] = request.POST[k]
                else:
                    print(f'missing value for measure key {k}')
            sub.measures_gt = cur_measures
            sub.save()
        return redirect('polls:subject-detail', subject_id=sub.id)
    else:
        subjects = Subject.objects.all()
        sub = subjects[0]
        return render(request, 'polls/subject_m_update.html', {'subject':sub})

def subject_list(request):
    subjects = Subject.objects.all()
    return render(request, 'polls/subject_list.html', {'subjects':subjects})

def add_subject_(request):
    if request.method == 'POST':
        file_f = request.FILES['img_front']
        file_s = request.FILES['img_side']
        gender = 0 if request.POST['gender'] == 'women' else 1
        height = float(request.POST['height'])
        gender = float(gender)

        img_f_bytes = file_f.read()
        img_s_bytes = file_s.read()
        img_f = Image.open(io.BytesIO(img_f_bytes))
        img_s = Image.open(io.BytesIO(img_s_bytes))
        path = Path('/media/D1/data_1/projects/Oh/codes/web_body/polls/static/polls/images/')
        path_f = path/'kori_f.jpeg'
        path_s = path/'kori_s.jpeg'
        img_f.save(path_f)
        img_s.save(path_s)

        measures, img_body_viz = get_mesurement(img_f_bytes, img_s_bytes, height=height, gender = gender)

        img_body  = Image.fromarray(img_body_viz)
        path_body = path/'kori_body.jpeg'
        img_body.save(path_body)
        context = {}
        context['img_f_src'] = 'kori_f.jpeg'
        context['img_s_src'] = 'kori_s.jpeg'
        context['img_body_src'] = 'kori_body.jpeg'
        context['measuredict'] = measures
        return render(request, 'polls/add_subject.html', context)
    else:
        return render(request, 'polls/add_subject.html')

class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    #context_object_name = 'latest_question_list'

    def get_queryset(self):
        """
        Return the last five published questions (not including those set to be
        published in the future).
        """
        return Subject.objects.all()

class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'

    def get_queryset(self):
        """
        Excludes any questions that aren't published yet.
        """
        return Question.objects.filter(pub_date__lte=timezone.now())


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/results.html'


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
