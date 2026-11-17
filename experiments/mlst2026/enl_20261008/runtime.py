"""Source-only numerical imports and pre-import frozen executable verification."""
import ast,hashlib,importlib.abc,importlib.machinery,importlib.metadata,os,stat,sys,sysconfig
from pathlib import Path
BOUND=None
ROOT=Path(__file__).resolve().parent.parents[2]

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def executable_files(sha,cached=None):
    paths=set()
    for distribution in importlib.metadata.distributions():
        for rel in distribution.files or []:
            if not (Path(rel).suffix in ['.py','.so','.dylib','.dll'] or '.so.' in Path(rel).name):continue
            p=Path(distribution.locate_file(rel)).resolve()
            if p.is_file():paths.add(p)
    base=Path(sysconfig.get_path('stdlib')).resolve()
    for directory,folders,names in os.walk(base):
        folders[:]=[name for name in folders if name not in ['site-packages','__pycache__']]
        for name in names:
            p=Path(directory)/name
            if p.suffix in ['.py','.so'] or '.so.' in p.name:paths.add(p.resolve())
    paths.add(Path(sys.executable).resolve())
    cached=cached or {}
    return {str(p):cached[str(p)] if str(p) in cached else sha(p) for p in sorted(paths)}

def verify_file(path):
    if BOUND is None:return
    p=str(Path(path).resolve())
    if p not in BOUND or file_sha(p)!=BOUND[p]:raise ImportError('unbound/changed numerical runtime executable: '+p)

def generated_source(fullname,path):
    """Authenticate only Torch's fixed non-scriptable bootstrap template, without execution."""
    if fullname!='_remote_module_non_scriptable':raise ImportError('unrecognized generated module')
    module=sys.modules.get('torch.distributed.nn.jit.instantiator')
    if module is None:raise ImportError('generated module has no authenticated instantiator')
    origin=Path(module.__file__).resolve();verify_file(origin)
    template=origin.parent/'templates'/'remote_module_template.py'
    template_bytes=template.read_bytes()
    if BOUND is None or BOUND.get(str(template.resolve()))!=hashlib.sha256(template_bytes).hexdigest():
        raise ImportError('unbound/changed generated-module template')
    # Literal extraction avoids executing a mutable imported template function.
    values={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(template_bytes).body
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)}
    substitutions=dict(assign_module_interface_cls='module_interface_cls = None',args='*args',kwargs='**kwargs',
                       arg_types='*args, **kwargs',arrow_and_return_type='',arrow_and_future_return_type='',jit_script_decorator='')
    expected=(values['_TEMPLATE_PREFIX']+values['_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA']).format(**substitutions).encode('utf-8')
    directory=Path(module.INSTANTIATED_TEMPLATE_DIR_PATH)
    p=Path(path)
    if not directory.is_absolute() or directory!=Path(module._TEMP_DIR.name) or directory.is_symlink() or p!=directory/(fullname+'.py'):
        raise ImportError('generated module origin mismatch')
    for parent in [directory,*directory.parents]:
        if parent.is_symlink():raise ImportError('generated module parent symlink refused')
    dfd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        ds=os.fstat(dfd)
        if ds.st_uid!=os.getuid() or stat.S_IMODE(ds.st_mode)!=0o700:raise ImportError('generated module directory ownership/mode')
        fd=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=dfd)
        with os.fdopen(fd,'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ImportError('generated module is not regular source')
            actual=stream.read(len(expected)+1)
    finally:os.close(dfd)
    if actual!=expected:raise ImportError('generated module template bytes differ')
    return actual

class GeneratedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname):
        # Compile the exact checked bytes; never reread a file or execute its cache.
        return self.source_to_code(generated_source(fullname,self.path),self.path)

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
            if not Path(spec.origin).resolve().is_relative_to(ROOT):
                if BOUND is not None and fullname=='_remote_module_non_scriptable':
                    generated_source(fullname,spec.origin);spec.loader=GeneratedLoader(fullname,spec.origin)
                else:
                    verify_file(spec.origin);spec.loader=SourceLoader(fullname,spec.origin)
        elif isinstance(spec.loader,importlib.machinery.ExtensionFileLoader):verify_file(spec.origin)
        elif spec.loader is not None:raise ImportError('unbound executable loader refused: '+type(spec.loader).__name__)
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
