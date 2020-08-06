import json, os

with open('pc_setup.json', 'rb') as f:
    setup = json.load(f)
which_pc = setup['which_pc']

if which_pc == 'local_gsa_paper':
    path_base = os.path.join('write_files', 'paper_gsa')
elif which_pc == 'merlin_gsa_paper':
    path_base = os.path.join('/','data','user','kim_a', 'paper_gsa')
