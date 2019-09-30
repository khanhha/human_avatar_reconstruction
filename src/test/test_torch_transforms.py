from torchvision import transforms
from pathlib import Path
import PIL.Image as Image
import os

if __name__ == '__main__':
    in_img_size = (224,224)
    img_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.RandomRotation(degrees=3),
        transforms.Resize(in_img_size)])

    dir = '/media/F/projects/Oh/data/cnn/sil_384_256_ml_fml_pose_nosyn_color/sil_f/test/'
    dir_out = f'/media/F/projects/Oh/data/cnn/sil_384_256_ml_fml_pose_nosyn_color/sil_f/tmp_test_debug/'
    os.makedirs(dir_out, exist_ok=True)
    paths = [path for path in Path(dir).glob('*.*')]
    for path in paths[:200]:
        img = Image.open(str(path))
        img_1 = img_transform(img)
        img_1.save(f'{dir_out}/{path.name}')

