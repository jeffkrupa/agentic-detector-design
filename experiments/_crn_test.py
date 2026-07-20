import numpy as np
import tools.sim as _simmod
from tools.sim import load_config, ctrl_flags_from_config, run_forward, default_design_point
_simmod._subprocess_timeout_s = lambda cfg=None: 1400.0
cfg=load_config(); ctrl=ctrl_flags_from_config(cfg)
def dp(a,g,e,p):
    d=default_design_point(cfg)
    return d.__class__(a=a,g=g,energy=e,n_layers=40,transverse=400.0,particle=p,abs_profile=None,gap_profile=None)
def tot(a,g,e,p,n,s,seeded="a"):
    rr=run_forward(dp(a,g,e,p),seeded,n_events=n,seed=s,ctrl=ctrl)
    assert rr.returncode==0 and rr.edeps is not None
    return float(np.asarray(rr.edeps)[:,0].sum())
N=2000
base=tot(3.5,5.7,10000.0,"e-",N,1)
hp=tot(3.55,5.7,10000.0,"e-",N,1)
hm=tot(3.45,5.7,10000.0,"e-",N,1)
print(f"N={N} s=1: total(base)={base:.2f} total(+.05)={hp:.2f} total(-.05)={hm:.2f}")
print(f"  FD d/da = {(hp-hm)/0.1:.4g}   (Delta={hp-hm:.4f})")
hp2=tot(3.55,5.7,10000.0,"e-",N,2)
hm2=tot(3.45,5.7,10000.0,"e-",N,2)
print(f"  s=2: FD d/da = {(hp2-hm2)/0.1:.4g}   (Delta={hp2-hm2:.4f})")
