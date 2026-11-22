"""Fix layout in manuscript figure copies while retaining the archived Slurm outputs."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib.pyplot as plt
import plot_paper_review as plots
from direction_pipeline import sha256

bundle=ROOT/'docs/paper/manuscript_v2'
provenance=json.loads((bundle/'assembly_provenance.json').read_text())
assets=bundle/'figures'
for name in ['F01_workflow','F03_architecture']:
    for suffix in ['pdf','png']:
        file=f'{name}.{suffix}'
        if sha256(assets/file)!=provenance['original_figures'][file]:
            raise ValueError('Figure copy already changed; preserve this version.')
original_save=plots.save

def save(fig,out,name):
    if name=='F01_workflow':
        ax=fig.axes[0]
        for text in ax.texts:
            if text.get_text().startswith('Direction is a completed ablation'):
                text.set_position((.02,-.10));text.set_va('top');text.set_fontsize(8)
        ax.set_xlim(-.03,1.03);ax.set_ylim(-.17,1.03)
    else:
        for ax in fig.axes:ax.set_xlim(-.03,1.03)
    original_save(fig,out,name)

plots.save=save
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
plots.workflow(assets);plots.architecture(assets)
corrections={}
for name in ['F01_workflow','F03_architecture']:
    for suffix in ['pdf','png']:
        file=f'{name}.{suffix}'
        corrections[file]=dict(original_sha256=provenance['original_figures'][file],manuscript_sha256=sha256(assets/file))
        shutil.copyfile(assets/file,ROOT/'figures'/file)
provenance['layout_corrections']=dict(reason='Separate workflow footer from validation box; preserve rounded node borders.',
    script_sha256=sha256(Path(__file__)),files=corrections,original_job_outputs_unchanged=True)
(bundle/'assembly_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Corrected four manuscript figure files; original figure job retained.')
