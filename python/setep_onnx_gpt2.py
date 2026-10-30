"""Explicit token-level adapter for models/chatgpt2/chatgpt2_impl.py.

This profile supplies opaque semantics; it is not inferred from a filename.
The SETEP computations still supply all projections, cache updates and loops.
"""
import json
import math
import numpy as np
from onnx import helper as h

from setep_onnx import Lowering, Value, require


class GPT2Lowering(Lowering):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.phase='prefill';self.iteration=0;self.schedule={};self.logits=[]
        self.mask=None;self.next_position=None

    def node(self,*args,**kwargs):
        value=super().node(*args,**kwargs)
        self.schedule[value.name]={'phase':self.phase,'iteration':self.iteration,'tokens':0}
        return value

    def parameter(self,key,shape,dtype):
        if key in self.used_weights:return self.used_weights[key]
        if key in self.weights:array=self.weights[key]
        else:
            require(self.placeholders,'Missing weight '+key+'; supply affine LN weights in the weight NPZ (--weights)')
            array=np.zeros(shape,dtype=dtype);self.placeholder_names.append(key)
        require(array.shape==shape and array.dtype==dtype,'Weight shape/dtype mismatch: '+key)
        value=self.constant(array,dtype,name=key);self.used_weights[key]=value
        return value

    def opaque(self,name,args,path,occurrence):
        if name=='LOOKUP':
            require(len(args)==2,'LOOKUP expects indices and a table')
            ids,table=args
            require(len(ids.shape)==1 and ids.dtype in (np.dtype('int32'),np.dtype('int64')) and len(table.shape)==2,
                    'LOOKUP expects integer indices and a rank-2 table')
            return self.node('Gather',[table,ids],ids.shape+table.shape[1:],table.dtype,axis=0)
        if name=='LN':
            require(len(args)==1,'LN expects one tensor')
            a=args[0]
            if '_BLOCK_' in occurrence:
                require(path and occurrence.endswith(('LN_0','LN_1')),'Unsupported block LN occurrence '+occurrence)
                key=f'h.{path[-1]}.ln_{1 if occurrence.endswith("LN_0") else 2}'
            else:
                require(occurrence in ('PREFILL_LN_0','DECODE_LN_0'),'Unsupported final LN occurrence '+occurrence)
                key='ln_f'
            normalized=self.operation('layernorm_last',[a])
            scale=self.parameter(key+'.weight',(a.shape[-1],),a.dtype)
            bias=self.parameter(key+'.bias',(a.shape[-1],),a.dtype)
            return self.binary('Add',self.binary('Mul',normalized,scale),bias)
        if name=='SOFTMAX':
            require(len(args)==2,'GPT-2 SOFTMAX expects scores and padding mask')
            scores,mask=args
            require(len(scores.shape)==2 and mask.shape==(scores.shape[-1],),'Attention mask shape mismatch')
            q,k=scores.shape
            scaled=self.binary('Div',scores,math.sqrt(self.scalars['model_width']//self.scalars['heads']))
            valid_mask=self.node('Cast',[mask],mask.shape,np.bool_,to=9)
            causal=self.constant(np.tri(q,k,k-q,dtype=bool),np.bool_)
            valid=self.node('And',[causal,valid_mask],scores.shape,np.bool_)
            numeric=self.node('Cast',[valid],scores.shape,scores.dtype,to=h.np_dtype_to_tensor_dtype(scores.dtype))
            nonempty=self.node('Greater',[self.reduce('ReduceSum',numeric),self.constant(0,scores.dtype)],(q,1),np.bool_)
            masked=self.node('Where',[valid,scaled,self.constant(-np.inf,scores.dtype)],scores.shape,scores.dtype)
            safe=self.node('Where',[nonempty,masked,self.constant(0,scores.dtype)],scores.shape,scores.dtype)
            probabilities=self.node('Softmax',[safe],scores.shape,scores.dtype,axis=-1)
            return self.node('Where',[valid,probabilities,self.constant(0,scores.dtype)],scores.shape,scores.dtype)
        if name=='SAMPLE':
            require(len(args)==1 and len(args[0].shape)==2,'SAMPLE expects sequence-by-vocabulary logits')
            a=args[0];self.logits.append(a)
            last=self.node('Gather',[a,self.constant(-1)],(a.shape[-1],),a.dtype,axis=0)
            token=self.node('ArgMax',[last],(1,),np.int64,axis=-1,keepdims=1)
            self.schedule[token.name]['tokens']=1
            return self.node('Cast',[token],(1,),np.int32,to=6)
        if name=='DECODE_PREP':
            require(len(args)==1 and args[0].shape==(1,) and self.mask is not None,'DECODE_PREP requires initialized prompt and one token')
            require(self.mask.shape[0]<self.scalars['positions'],'Decode exceeds position capacity')
            self.mask=self.concat([self.mask,self.constant([1],np.int32)],axis=0)
            position=self.next_position
            self.next_position=self.binary('Add',position,1)
            return {'TOK':args[0],'MASK':self.mask,'POSITION':position}
        # Only the documented activation contract is inherited. Unknown abstract
        # operations remain errors, including text tokenizer/detokenizer calls.
        require(name=='GELU','Unsupported GPT-2 opaque operation '+name)
        return self.operation(name,args)

    def run(self):
        require({'PREFILL','DECODE'}<=self.functions.keys(),'gpt2-token requires PREFILL and DECODE computations')
        for key in ('tokens','decode_tokens','positions','model_width','heads','layers'):
            require(key in self.scalars,'gpt2-token requires scalar '+key)
        tokens=self.scalars['tokens'];count=self.scalars['decode_tokens']
        require(tokens>0 and 0<=count<=1000 and tokens+count<=self.scalars['positions'],'Invalid GPT-2 prompt/decode lengths')
        require(self.scalars['heads']>0 and self.scalars['model_width']%self.scalars['heads']==0,'Invalid head width')
        def input(name):
            self.inputs.append(h.make_tensor_value_info(name,6,[tokens]))
            return Value(name,(tokens,),np.dtype('int32'))
        ids=input('input_ids');self.mask=input('attention_mask')
        cumulative=self.node('CumSum',[self.mask,self.constant(0)],(tokens,),np.int32)
        positions=self.binary('Sub',cumulative,1)
        positions=self.node('Max',[positions,self.constant(0,np.int32)],(tokens,),np.int32)
        self.next_position=self.node('ReduceSum',[self.mask,self.constant([0])],(1,),np.int32,keepdims=1)
        def invoke(name,inputs):
            f=self.functions[name]
            require(len(f['inputs'])==len(inputs) and not f['needs_instance'],'Unexpected '+name+' signature')
            return self.call(f,inputs+[(),()])
        generated=[invoke('PREFILL',[ids,self.mask,positions])]
        for i in range(count):
            self.phase='decode';self.iteration=i+1
            generated.append(invoke('DECODE',[generated[-1]]))
        result={'tokens':self.concat(generated,axis=0),'logits':self.logits}
        model=self.finish(result,'GPT2_TOKEN_GENERATION',{
            'setep.export_profile':'gpt2-token',
            'setep.math_conventions':'chatgpt2_impl.py: affine LN epsilon=1e-5; scaled causal/padding attention; empty rows zero; tanh GELU; greedy last-token sampling',
            'setep.token_contract':'batch=1; fixed int32 input_ids and binary left-padding attention_mask; at least one real token; external tokenizer/detokenizer',
            'setep.schedule':json.dumps({'version':1,'prompt_tokens':tokens,'output_tokens':count+1,'nodes':self.schedule})})
        return model
