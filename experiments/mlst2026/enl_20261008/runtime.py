"""Source-only numerical imports and pre-import frozen executable verification."""
import importlib.abc,importlib.machinery,importlib.metadata,sys,sysconfig
from pathlib import Path
BOUND=None

def executable_files(sha):
    paths=set()
    for distribution in importlib.metadata.distributions():
        for rel in distribution.files or []:
            p=Path(distribution.locate_file(rel)).resolve()
            if p.is_file() and (p.suffix in ['.py','.so','.dylib','.dll'] or '.so.' in p.name):paths.add(p)
    base=Path(sysconfig.get_path('stdlib')).resolve()
    for p in base.rglob('*'):
        if 'site-packages' in p.parts:continue
        if p.is_file() and (p.suffix in ['.py','.so'] or '.so.' in p.name):paths.add(p.resolve())
    paths.add(Path(sys.executable).resolve())
    return {str(p):sha(p) for p in sorted(paths)}

def verify_file(path):
    if BOUND is None:return
    from .workflow import sha
    p=str(Path(path).resolve())
    if p not in BOUND or sha(p)!=BOUND[p]:raise ImportError('unbound/changed numerical runtime executable: '+p)

class SourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname):
        verify_file(self.path)
        return self.source_to_code(self.get_data(self.path),self.path)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is None:return None
        if isinstance(spec.loader,importlib.machinery.SourcelessFileLoader):raise ImportError('sourceless executable cache refused')
        if isinstance(spec.loader,importlib.machinery.SourceFileLoader):
            # Repository source is bound separately; numerical dependency source bypasses .pyc.
            from .workflow import ROOT
            if not Path(spec.origin).resolve().is_relative_to(ROOT):
                verify_file(spec.origin);spec.loader=SourceLoader(fullname,spec.origin)
        elif isinstance(spec.loader,importlib.machinery.ExtensionFileLoader):verify_file(spec.origin)
        return spec

def install(bound=None):
    global BOUND
    BOUND=bound
    if not any(isinstance(x,Finder) for x in sys.meta_path):sys.meta_path.insert(0,Finder())

def preimport(environment,sha):
    bound=environment.get('executable_files')
    if not isinstance(bound,dict) or not bound or environment.get('source_only_imports') is not True:raise ValueError('frozen executable closure missing')
    if executable_files(sha)!=bound:raise ValueError('pre-import runtime executable census/hash mismatch')
    install(bound)
