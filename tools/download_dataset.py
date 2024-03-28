from zenodo_get import zenodo_get
from tqdm import tqdm
import zipfile

SENSORY_ART_RECORD = '10889613'

def main():
    zenodo_get([SENSORY_ART_RECORD, '-odata'])

    for f in tqdm(['annotations.zip', 'images.zip']):
        with zipfile.ZipFile(f'data/{f}') as zf:
            zf.extractall('data/')

if __name__ == '__main__':
    main()