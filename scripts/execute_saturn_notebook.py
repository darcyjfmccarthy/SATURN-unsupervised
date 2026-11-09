#!/usr/bin/env python3
"""Execute with visible streams and an atomic notebook snapshot after each cell."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jax_saturn.progress import Progress


def execute(notebook, output, *, env=None):
    import psutil
    existing_children = {child.pid for child in psutil.Process().children(recursive=True)}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    nb = nbformat.read(notebook,as_version=4)
    # Source notebooks may have old interactive outputs; execution always begins
    # with a clean source and publishes each newly completed cell.
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs=[]
            cell.execution_count=None
    def save():
        staging = output.with_name(output.name+'.tmp')
        nbformat.write(nb,staging)
        staging.replace(output)
    with Progress(output.parent/'notebook-execution.jsonl') as progress:
        class StreamingClient(NotebookClient):
            def process_message(self,msg,cell,cell_index):
                if msg['msg_type']=='stream':
                    target = sys.stderr if msg['content'].get('name')=='stderr' else sys.stdout
                    print(msg['content'].get('text',''),end='',file=target,flush=True)
                elif msg['msg_type']=='error':
                    print('\n'.join(msg['content'].get('traceback',[])),file=sys.stderr,flush=True)
                return super().process_message(msg,cell,cell_index)
        def started(cell,cell_index,**kwargs):
            progress.active=f'notebook_cell_{cell_index}'
            progress.event('cell_start',cell=cell_index)
        def completed(cell,cell_index,**kwargs):
            save()
            progress.event('cell_complete',cell=cell_index)
        with tempfile.TemporaryDirectory(prefix='saturn-kernel-') as temporary:
            # Kernel uses the invoking environment, not an arbitrary globally
            # registered python3 kernel or the user's default conda environment.
            kernel = Path(temporary)/'kernels/saturn-runner'
            kernel.mkdir(parents=True)
            import json
            (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                'display_name':'SATURN runner','language':'python'}))
            old = os.environ.get('JUPYTER_PATH')
            os.environ['JUPYTER_PATH']=temporary+(os.pathsep+old if old else '')
            client=StreamingClient(nb,timeout=None,kernel_name='saturn-runner',resources={'metadata':{'path':str(ROOT)}},
                on_cell_start=started,on_cell_executed=completed,allow_errors=False)
            try:
                save()
                client.execute(env=env or os.environ.copy())
            finally:
                save()  # includes the failing cell's traceback and prior outputs
                if old is None:
                    os.environ.pop('JUPYTER_PATH',None)
                else:
                    os.environ['JUPYTER_PATH']=old
                # NotebookClient's kernel context terminates on success/error.
                # Terminate shell children too (e.g. evaluation interrupted mid-cell).
                children=[child for child in psutil.Process().children(recursive=True) if child.pid not in existing_children]
                for child in reversed(children):
                    try: child.terminate()
                    except psutil.NoSuchProcess: pass
                _,alive=psutil.wait_procs(children,timeout=2)
                for child in alive:
                    try: child.kill()
                    except psutil.NoSuchProcess: pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    execute(args.notebook,args.output)

if __name__=='__main__':
    main()
