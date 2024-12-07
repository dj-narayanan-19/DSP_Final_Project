import numpy as np
import requests
import os

query_params = {
    'grp' : 'birds',
    'cnt' : 'united_states',
    'stage' : 'adult'
}
query = '+'.join([param + ':' + query_params[param] for param in query_params])

api = f'https://xeno-canto.org/api/2/recordings?query={query}'
file_data = requests.get(api).json()
num_pages = np.ceil(int(file_data['numRecordings'])/500).astype(int)

save_dir = '../data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
seen_animals = set([animal.split('.')[0] for animal in os.listdir(save_dir)])

for i in range(1, num_pages+1):
    print(f'On Page {i} of {num_pages+1}')
    api = f'https://xeno-canto.org/api/2/recordings?query={query}&page={i}'
    r = requests.get(api)
    file_data = r.json()
    recordings = sorted(file_data['recordings'], key= lambda x: 60*int(x['length'].split(':')[0])+int(x['length'].split(':')[1]))
    for recording in recordings:
        animal = recording['gen'] + '_' + recording['sp']
        if animal not in seen_animals:
            try:
                with requests.get(recording['file'], allow_redirects=True, stream=True) as r:
                    with open(f"{save_dir}/{animal}.mp3", "wb") as f:
                        f.write(r.content)
                        seen_animals.add(animal)
            except:
                print(f'no data for {animal}')
