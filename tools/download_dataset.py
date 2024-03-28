from zenodo_get import zenodo_get

SENSORY_ART_RECORD = '10889613'

def main():
    zenodo_get([SENSORY_ART_RECORD, '-o data'])

if __name__ == '__main__':
    main()