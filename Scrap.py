from Importer import Importer

im = Importer()

for t in im.get_test_set()['t']:
    print(t)