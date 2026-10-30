"""Executable ONNX lowering of the supported SETEP semantic IR subset.

The setep command line tool writes the semantic module as a JSON request and
runs this file:  python setep_onnx.py --request REQUEST.json
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import sys

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()

DTYPES={'fp16':np.float16,'fp32':np.float32,'fp64':np.float64,
        'int8':np.int8,'int16':np.int16,'int32':np.int32,'int64':np.int64,
        'uint8':np.uint8,'uint16':np.uint16,'uint32':np.uint32,'uint64':np.uint64,'bool':np.bool_}


@dataclass
class Value:
    name: str
    shape: tuple
    dtype: np.dtype


class Lowering:
    def __init__(self, module, weights=None, placeholders=False):
        self.m=module;self.weights=weights if weights is not None else {}
        self.placeholders=placeholders;self.placeholder_names=[];self.used_weights={}
        self.nodes=[];self.initializers=[];self.inputs=[];self.serial=0;self.depth=0
        self.functions={f['name']:f for f in module['functions']}
        self.scalars={k:int(v) for k,v in module['scalars'].items()}
        for i,f in enumerate(module['functions']):self.scalars[f['name']+'_family']=i
        self.states={};self.state_values=set()

    def fresh(self, label='v'):
        self.serial+=1
        require(self.serial<=200000,'ONNX export exceeds 200000 values; reduce repetition counts')
        return label+'_'+str(self.serial)

    def constant(self, value, dtype=np.int64, name=None):
        array=np.asarray(value,dtype=dtype)
        name=name or self.fresh('constant')
        self.initializers.append(nh.from_array(array,name))
        return Value(name,array.shape,array.dtype)

    def node(self, op, args, shape, dtype=None, **attrs):
        require(all(isinstance(a,Value) for a in args),op+' requires tensor operands')
        name=self.fresh(op)
        self.nodes.append(h.make_node(op,[a.name for a in args],[name],name=name,**attrs))
        return Value(name,tuple(shape),np.dtype(dtype or args[0].dtype))

    def binary(self, op, a, b):
        if not isinstance(a,Value):a=self.constant(a,b.dtype)
        if not isinstance(b,Value):b=self.constant(b,a.dtype)
        require(a.dtype==b.dtype,op+' operand dtypes differ')
        shape=np.broadcast_shapes(a.shape,b.shape)
        return self.node(op,[a,b],shape)

    def matmul(self,a,b):
        require(a.dtype==b.dtype and a.shape and b.shape,'MatMul rank/dtype mismatch')
        l=(1,)+a.shape if len(a.shape)==1 else a.shape
        r=b.shape+(1,) if len(b.shape)==1 else b.shape
        require(l[-1]==r[-2],'MatMul reduction dimensions differ')
        shape=np.broadcast_shapes(l[:-2],r[:-2])+(l[-2],r[-1])
        if len(a.shape)==1:shape=shape[:-2]+shape[-1:]
        if len(b.shape)==1:shape=shape[:-1]
        return self.node('MatMul',[a,b],shape)

    def reduce(self, op, a):
        require(a.shape,'last-axis reduction needs a non-scalar tensor')
        args=[a]
        attrs=dict(keepdims=1)
        if op=='ReduceSum':args.append(self.constant([-1]))
        else:attrs['axes']=[-1]
        return self.node(op,args,a.shape[:-1]+(1,),**attrs)

    def operation(self, op, args):
        # Defined conventions: last-axis normalization, epsilon 1e-5,
        # non-affine LN/RMSNorm, tanh-approximate GELU.
        aliases={'LN':'layernorm_last','LAYERNORM':'layernorm_last','RMSNORM':'rmsnorm_last',
                 'SOFTMAX':'softmax_last','MUL_ELEMENTS':'mul','DIV_ELEMENTS':'div'}
        op=aliases.get(op,op)
        require(len(args)==(2 if op in ('mul','div','maximum') else 1),'operation input arity: '+op)
        a=args[0]
        require(isinstance(a,Value),'operation requires tensor: '+op)
        if op in ('mul','div','maximum'):return self.binary({'mul':'Mul','div':'Div','maximum':'Max'}[op],*args)
        require(a.dtype.kind=='f','floating-point operation requires float input: '+op)
        if op in ('sum_last','mean_last','max_last'):
            return self.reduce({'sum_last':'ReduceSum','mean_last':'ReduceMean','max_last':'ReduceMax'}[op],a)
        if op=='softmax_last':return self.node('Softmax',[a],a.shape,axis=-1)
        if op in ('layernorm_last','rmsnorm_last'):
            centered=self.binary('Sub',a,self.reduce('ReduceMean',a)) if op=='layernorm_last' else a
            variance=self.reduce('ReduceMean',self.binary('Mul',centered,centered))
            root=self.node('Sqrt',[self.binary('Add',variance,1e-5)],variance.shape)
            return self.binary('Div',centered,root)
        if op in ('RELU','SIGMOID','TANH','exp'):
            return self.node({'RELU':'Relu','SIGMOID':'Sigmoid','TANH':'Tanh','exp':'Exp'}[op],[a],a.shape)
        if op=='SILU':return self.binary('Mul',a,self.node('Sigmoid',[a],a.shape))
        if op=='GELU':
            cube=self.binary('Mul',self.binary('Mul',a,a),a)
            inner=self.binary('Mul',self.binary('Add',a,self.binary('Mul',cube,0.044715)),math.sqrt(2/math.pi))
            return self.binary('Mul',self.binary('Mul',a,0.5),self.binary('Add',self.node('Tanh',[inner],a.shape),1.0))
        if op=='rsqrt_eps':return self.binary('Div',1.0,self.node('Sqrt',[self.binary('Add',a,1e-5)],a.shape))
        if op.startswith('divide_by:'):return self.binary('Div',a,float(op.split(':')[1]))
        raise ValueError('No mathematical ONNX implementation for abstract computation '+op)

    def local(self, name, env):
        if name in env:return env[name]
        if name in self.scalars:return self.scalars[name]
        try:return int(name)
        except ValueError:raise ValueError('Unresolved scalar/local '+name)

    def shape(self, dims):return tuple(self.local(str(d),{}) for d in dims)

    def weight(self, bank, path):
        require(isinstance(path,tuple),'weight path must be a static instance path')
        key=bank+('/'+'/'.join(map(str,path)) if path else '')
        if key in self.used_weights:return self.used_weights[key]
        declaration=self.m['weight_declarations'].get(bank)
        require(declaration is not None,'weight has no declared shape: '+bank)
        shape=self.shape(self.m['weight_shapes'][declaration]);dtype=np.dtype(DTYPES[self.m['weight_dtypes'][declaration]])
        require(math.prod(shape)*dtype.itemsize<2**31,'weight exceeds embedded ONNX export limit')
        # Explicit bank/path keys are unambiguous. A declaration alias is only
        # accepted for root ownership, never silently shared across instances.
        alias=declaration if not path else key
        if key in self.weights:array=self.weights[key]
        elif alias in self.weights:array=self.weights[alias]
        else:
            require(self.placeholders,'Missing weight '+key+'; supply a weight NPZ (--weights) or explicitly request zero placeholders (--zero-weights)')
            array=np.zeros(shape,dtype=dtype);self.placeholder_names.append(key)
        require(array.shape==shape and array.dtype==dtype,'Weight shape/dtype mismatch: '+key)
        value=self.constant(array,dtype,name=key);self.used_weights[key]=value
        return value

    def concat(self,args,axis=-1):
        require(args,'empty tensor concatenation')
        shape=list(args[0].shape);axis%=len(shape);shape[axis]=0
        for a in args:
            require(a.dtype==args[0].dtype and len(a.shape)==len(shape) and
                    all(d==args[0].shape[i] for i,d in enumerate(a.shape) if i!=axis),'Concat shape/dtype mismatch')
            shape[axis]+=a.shape[axis]
        return self.node('Concat',args,shape,axis=axis)

    def split(self,a,count,index,axis=-1):
        require(count>0 and 0<=index<count and a.shape[axis]%count==0,'Invalid static split')
        width=a.shape[axis]//count;shape=list(a.shape);shape[axis]=width
        return self.node('Slice',[a,self.constant([index*width]),self.constant([(index+1)*width]),self.constant([axis])],shape)

    def expression(self,e,env):
        kind=e['kind'];name=e['name']
        if kind=='Local':return self.local(name,env)
        if kind=='StringLiteral':return name
        args=[self.expression(a,env) for a in e['args']]
        if kind=='IndexedPath':return () if not args else args[0]+tuple(args[1:])
        if kind=='Weight':return self.weight(name,args[0])
        if kind in ('StateRead','StateWrite'):
            require(isinstance(args[0],tuple),'State path must be static')
            key=(name,args[0])
            if kind=='StateWrite':
                require(isinstance(args[-1],Value),'Persistent state must be a tensor')
                self.states[key]=args[-1]
                self.state_values.add(args[-1].name)
            require(key in self.states,'Uninitialized persistent state '+str(key)+'; initialize it in the exported entry computation')
            return self.states[key]
        if kind=='StateConcat':return self.concat(args,axis=0)
        if kind=='Call':return self.call(self.functions[name],args)
        if kind=='Opaque':return self.opaque(name,args[:-2],args[-2],args[-1])
        if kind=='TensorOp':return self.operation(name,args)
        if kind in ('Add','Subtract','Scale'):return self.binary({'Add':'Add','Subtract':'Sub','Scale':'Mul'}[kind],*args)
        if kind=='Matmul':return self.matmul(*args)
        if kind=='Transpose':
            a=args[0];require(len(a.shape)>=2,'Transpose requires rank >= 2')
            perm=list(range(len(a.shape)));perm[-1],perm[-2]=perm[-2],perm[-1]
            return self.node('Transpose',[a],tuple(a.shape[i] for i in perm),perm=perm)
        if kind=='ValidateShape':
            require(isinstance(args[0],Value) and args[0].shape==tuple(args[1:]),'Declared tensor shape mismatch')
            return args[0]
        if kind=='Array':return args[0] if name else args
        if kind=='ArraySize':return len(args[0])
        if kind in ('ArrayAt','GroupMember'):
            if kind=='GroupMember':require(len(args[0])==args[1],'Group arity mismatch')
            return args[0][args[-1]]
        if kind=='ValidateArray':
            require(isinstance(args[0],list) and len(args[0])==args[1],'Array cardinality mismatch')
            return args[0]
        if kind=='Group':return args if name=='literal' else args[0]
        if kind=='Record':return dict(zip(self.m['input_records'][name],args))
        if kind=='Member':return args[0][name]
        if kind=='Split':return self.split(*args,axis=int(name) if name else -1)
        if kind=='Merge':
            if name=='sum':
                require(args[0],'Empty reduction');out=args[0][0]
                for a in args[0][1:]:out=self.binary('Add',out,a)
                return out
            return self.concat(args[0] if name=='family' or name.startswith('axis:') else args,
                               int(name.split(':')[1]) if name.startswith('axis:') else -1)
        raise ValueError('Unsupported SETEP expression for ONNX export: '+kind)

    def opaque(self,name,args,path,occurrence):
        return self.operation(name,args)

    def statements(self,statements,env):
        for s in statements:
            kind=s['kind'];name=s['name']
            if kind=='Loop':
                require(not s['selector'],'Dynamic selection cannot be statically exported')
                count=self.local(s['count'],env);require(isinstance(count,int) and 0<=count<=100000,'Invalid loop count')
                for i in (reversed(range(count)) if s['reverse'] else range(count)):
                    env[name]=i;self.statements(s['body'],env)
            elif kind=='Assign':
                env[name]=[] if s['group'] else self.expression(s['value'],env)
                if s['declaration']:
                    value=env[name];decl=s['declaration']
                    require(isinstance(value,Value) and value.shape==self.shape(self.m['tensor_shapes'][decl]),'Declared assignment shape mismatch: '+decl)
                    require(value.dtype==np.dtype(DTYPES[self.m['tensor_dtypes'][decl]]),'Declared assignment dtype mismatch: '+decl)
            elif kind in ('Append','Extend'):
                value=self.expression(s['value'],env)
                if kind=='Extend':env[name].extend(value)
                elif s['reverse']:env[name].insert(0,value)
                else:env[name].append(value)
            elif kind=='Check':
                value=self.expression(s['value'],env)
                require(isinstance(value,(bool,int)),'Dynamic SETEP check cannot be statically exported: '+name)
                require(bool(value),name)
            else:raise ValueError('Unsupported SETEP statement '+kind)

    def call(self,f,args):
        self.depth+=1;require(self.depth<=100,'Recursive/excessively nested SETEP computations')
        names=f['inputs']+['instance_path']+(['instance'] if f['needs_instance'] else [])+['state_path']
        require(len(names)==len(args),'Function arity mismatch: '+f['name'])
        env=dict(zip(names,args));env.setdefault('instance',0)
        self.statements(f['body'],env)
        values=[self.expression(e,env) for e in f['results']]
        self.depth-=1
        if f['returns_group']:return values
        return values[0] if len(values)==1 else dict(zip(f['outputs'],values))

    def run(self):
        f=self.m['functions'][-1];args=[]
        for name in f['inputs']:
            if not f['consumes_input']:
                args.append(None)
                continue
            require(not f['input_types'].get(name),'Structured/array entry input requires explicit flattening: '+name)
            decl=f['input_declarations'].get(name,name)
            require(decl in self.m['tensor_shapes'],'Entry input requires a declared shape: '+name)
            shape=self.shape(self.m['tensor_shapes'][decl]);dtype=np.dtype(DTYPES[self.m['tensor_dtypes'][decl]])
            value=Value(name,shape,dtype);args.append(value)
            self.inputs.append(h.make_tensor_value_info(name,h.np_dtype_to_tensor_dtype(dtype),shape))
        result=self.call(f,args+[()]+([0] if f['needs_instance'] else [])+[()])
        return self.finish(result,f['name'])

    def finish(self,result,name,properties=None):
        outputs=[]
        def output(value,label):
            if isinstance(value,dict):
                for key,v in value.items():output(v,label+'_'+key)
            elif isinstance(value,list):
                for i,v in enumerate(value):output(v,label+'_'+str(i))
            else:
                require(isinstance(value,Value),'ONNX output must be a tensor')
                alias=self.fresh(label);self.nodes.append(h.make_node('Identity',[value.name],[alias]))
                outputs.append(h.make_tensor_value_info(alias,h.np_dtype_to_tensor_dtype(value.dtype),value.shape))
        output(result,'output')
        for (bank,path),value in sorted(self.states.items()):
            output(value,'state_'+bank+''.join('_'+str(i) for i in path))
        model=h.make_model(h.make_graph(self.nodes,name,self.inputs,outputs,self.initializers),
                           opset_imports=[h.make_opsetid('',13)],ir_version=9,producer_name='setep')
        h.set_model_props(model,{'setep.math_conventions':'last-axis non-affine LN/RMSNorm epsilon=1e-5; GELU=tanh approximation; static repetition expanded',
                                'setep.state_contract':'state initialized within entry; static SSA updates; final state outputs',
                                'setep.cache_values':json.dumps(sorted(self.state_values)),**(properties or {})})
        onnx.checker.check_model(model)
        return onnx.shape_inference.infer_shapes(model,strict_mode=True)


def publish(model,source,target=None,overwrite=False,placeholders=None):
    """Publish a checked embedded-data model with an ownership/checksum sidecar."""
    source=Path(source);target=Path(target) if target else source.with_suffix('.onnx')
    receipt=target.with_suffix('.onnx-export.json')
    require(not target.is_symlink() and not receipt.is_symlink(),'Refusing symbolic-link export targets')
    existing=target.exists() or receipt.exists()
    require(not existing or overwrite,'ONNX export collision: '+str(target))
    if existing:
        require(receipt.is_file(),'Refusing to overwrite an ONNX model without a generated ownership receipt')
        prior=json.loads(receipt.read_text())
        require(prior.get('format')=='setep.onnx-export' and prior.get('version')==1 and prior.get('source_file')==source.name and
                prior.get('model_file')==target.name,'ONNX target is not owned by this SETEP export')
        if target.exists():
            with target.open('rb') as stream:require(hashlib.file_digest(stream,'sha256').hexdigest()==prior['model_sha256'],'Existing ONNX was modified; refusing overwrite')
    require(model.ByteSize()<2**31,'Embedded ONNX export exceeds 2 GiB; external-data export is not implemented')
    props={p.key:p.value for p in model.metadata_props}
    if placeholders is None:
        placeholders=json.loads(props.get('setep.placeholder_weights','[]')) if props.get('setep.weight_status')=='placeholder' else []
        require(isinstance(placeholders,list) and all(isinstance(p,str) for p in placeholders),
                'Invalid preserved placeholder provenance')
        require(props.get('setep.weight_status')!='placeholder' or placeholders,
                'Placeholder model lacks preserved weight identities')
    props.update({'setep.weight_status':'placeholder' if placeholders else 'supplied',
                  'setep.source_sha256':digest(source.read_bytes()),
                  'setep.placeholder_weights':json.dumps(list(placeholders))})
    h.set_model_props(model,props)
    onnx.checker.check_model(model)
    tag='.tmp-'+uuid.uuid4().hex;temps=[Path(str(p)+tag) for p in (target,receipt)]
    backups={};published=[]
    try:
        with temps[0].open('xb') as stream:stream.write(model.SerializeToString())
        with temps[0].open('rb') as stream:model_hash=hashlib.file_digest(stream,'sha256').hexdigest()
        data=dict(format='setep.onnx-export',version=1,source_file=source.name,model_file=target.name,
                  model_sha256=model_hash,source_sha256=digest(source.read_bytes()),placeholder_weights=list(placeholders))
        with temps[1].open('x',encoding='utf-8') as stream:json.dump(data,stream,indent=2);stream.write('\n')
        for p in (target,receipt):
            if overwrite and p.exists():
                backup=Path(str(p)+tag+'.previous');os.replace(p,backup);backups[p]=backup
        for temp,p in zip(temps,(target,receipt)):
            if overwrite:os.replace(temp,p)
            else:os.link(temp,p);temp.unlink()
            published.append(p)
    except Exception:
        for p in published:p.unlink(missing_ok=True)
        for p,backup in backups.items():os.replace(backup,p)
        raise
    finally:
        for temp in temps:temp.unlink(missing_ok=True)
    for backup in backups.values():backup.unlink()
    return {'schema_version':1,'onnx_file':str(target),'placeholder_weights':list(placeholders)}


def export_native(request):
    source=Path(request['model']);weights_path=request.get('weights','')
    target=Path(request['output']) if request.get('output') else source.with_suffix('.onnx')
    path=Path(weights_path) if weights_path else source.with_suffix('.weights.npz')
    require(not weights_path or path.is_file(),'Weight NPZ not found: '+str(path))
    weights=np.load(path,allow_pickle=False) if path.exists() else None
    try:
        profile=request.get('profile','')
        require(profile in ('','gpt2-token'),'Unknown ONNX export profile: '+profile)
        cls=Lowering
        if profile=='gpt2-token':
            from setep_onnx_gpt2 import GPT2Lowering
            cls=GPT2Lowering
        lowering=cls(request['module'],weights,request.get('placeholder_weights',False))
        model=lowering.run()
        return publish(model,source,target,request.get('overwrite_generated',False),lowering.placeholder_names)
    finally:
        if weights is not None:weights.close()


def main():
    parser=argparse.ArgumentParser(description='Export a SETEP semantic module request to ONNX')
    parser.add_argument('--request',required=True,help='JSON request written by the setep tool')
    args=parser.parse_args()
    try:
        request=json.loads(Path(args.request).read_text(encoding='utf-8'))
        require(request.get('operation')=='setep_to_onnx','Unsupported request operation')
        result=export_native(request)
    except Exception as error:
        print('setep: ONNX export: '+str(error),file=sys.stderr)
        return 1
    print(result['onnx_file'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
