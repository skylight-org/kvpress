import sys, torch
sys.path.insert(0,'.')
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
MID="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(MID)
model=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
dev=model.device; nkv=model.config.num_key_value_heads
ids=tok("The capital of France is Paris. "*200,return_tensors="pt").input_ids.to(dev); n=ids.shape[1]
q=tok(" What is the capital?",return_tensors="pt",add_special_tokens=False).input_ids.to(dev)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
ORIG=ALL_ATTENTION_FUNCTIONS["sdpa"]
info={}
def make(mode):
    def f(module,query,key,value,attention_mask,dropout=0.0,**kw):
        bias=getattr(module,"e2_bias",None)
        if bias is not None and query.shape[2]!=key.shape[2]:
            nq,nk=query.shape[2],key.shape[2]; nkv_,n_ctx=bias.shape[1],bias.shape[2]
            G=query.shape[1]//nkv_
            add=bias.repeat_interleave(G,dim=1)[:,:,None,:]
            if mode=="rows":   full=torch.zeros((1,query.shape[1],nq,nk),dtype=query.dtype,device=query.device)
            else:              full=torch.zeros((1,query.shape[1],1,nk),dtype=query.dtype,device=query.device)
            full[...,:n_ctx]=add.to(query.dtype)
            info.setdefault(mode,{"mask_none":attention_mask is None,"mdtype":str(None if attention_mask is None else attention_mask.dtype),"mmin":float("nan") if attention_mask is None else float(attention_mask.min()),
                                  'mshape':None if attention_mask is None else tuple(attention_mask.shape),
                                  'fullshape':tuple(full.shape)})
            attention_mask = full if attention_mask is None else attention_mask+full
        return ORIG(module,query,key,value,attention_mask,dropout,**kw)
    return f
def run(set_fn,mode=None):
    if mode: ALL_ATTENTION_FUNCTIONS["sdpa"]=make(mode)
    cache=DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ids,past_key_values=cache); set_fn(cache)
        out=model(input_ids=q,past_key_values=cache,position_ids=torch.arange(n,n+q.shape[1],device=dev)[None]).logits[0].double()
    for L in model.model.layers: L.self_attn.e2_bias=None; L.self_attn.masked_key_indices=None
    ALL_ATTENTION_FUNCTIONS["sdpa"]=ORIG
    return out
zero=lambda c:[setattr(L.self_attn,"e2_bias",torch.zeros(1,nkv,n,device=dev)) for L in model.model.layers]
base=run(lambda c:None)
for mode in ("bcast","rows"):
    z=run(zero,mode); print(f"{mode:6s} T1 rel {((z-base).abs().max()/base.abs().max()).item():.6f}  {info.get(mode)}")
