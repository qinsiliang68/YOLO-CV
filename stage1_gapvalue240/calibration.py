from __future__ import annotations
from dataclasses import asdict,dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression

@dataclass(frozen=True)
class PlattModel:
    coefficient:float
    intercept:float
    source_prevalence:float
    deployment_prevalence:float
    clip_low:float=1e-7
    clip_high:float=.9999999

    def transform(self,p:np.ndarray)->np.ndarray:
        p=np.clip(np.asarray(p,dtype=np.float64),self.clip_low,self.clip_high)
        logit=np.log(p/(1-p)); z=self.coefficient*logit+self.intercept
        source=np.clip(self.source_prevalence,self.clip_low,self.clip_high)
        deploy=np.clip(self.deployment_prevalence,self.clip_low,self.clip_high)
        z += np.log(deploy/(1-deploy))-np.log(source/(1-source))
        return 1/(1+np.exp(-np.clip(z,-60,60)))

    def to_dict(self): return asdict(self)


def fit_platt(raw_probability:np.ndarray,y_true:np.ndarray,deployment_prevalence:float=.5)->PlattModel:
    p=np.clip(np.asarray(raw_probability,dtype=np.float64),1e-7,.9999999); y=np.asarray(y_true,dtype=np.int8)
    x=np.log(p/(1-p)).reshape(-1,1)
    model=LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000,random_state=0).fit(x,y)
    return PlattModel(float(model.coef_[0,0]),float(model.intercept_[0]),float(y.mean()),float(deployment_prevalence))
