import pickle

path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/debug/caesar_obj_gender.pkl'
OUT_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/debug/'
males = []
females = []
with open(path, 'rb') as file:
    genders = pickle.load(file)
    for name, gender in genders.items():
        if gender == True:
            females.append(name)
        else:
            males.append(name)

with open(f'{OUT_DIR}/males.pkl', 'wb') as file:
    pickle.dump(males, file)

with open(f'{OUT_DIR}/females.pkl', 'wb') as file:
    pickle.dump(females, file= file)

print('females: ', len(females))
print('males: ', len(males))
