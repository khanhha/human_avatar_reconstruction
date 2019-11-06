from django.template.defaultfilters import register
#from common.viz_util import measure_color_defs

@register.filter(name='dict_key')
def dict_key(d, k):
    return d[k]

@register.filter(name='measure_color')
def measure_color(k):
    cnames = measure_color_defs()
    if k in cnames:
        return cnames[k]
    else:
        #default color
        return 'gray'

@register.filter(name='err_str')
def err_str(gt, pred):
    gt = float(gt)
    pred = float(pred)
    if gt > 0.0 and pred >= 0.0:
        e = abs(gt-pred)/gt
        ret = f'{100*e:.2f}%'
        return ret
    else:
        return ''
