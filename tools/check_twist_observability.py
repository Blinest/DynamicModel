import sys, numpy as np
sys.path[:0]=['.','geometry','stiffness','force','plot']
import cosserat_lattice as CL
PSI=np.radians([0,60,120,180,240,300.]); T0=10.; r=5e-3; EI=CL.arrays()["EI"]
N=20; ds=0.4/N; ns=3
def forward(A, phi0_deg, tw_deg):
    p=np.zeros(3); R=np.eye(3); kap_acc=0.; P=[p.copy()]; Rc=[R.copy()]; s_all=[0.]
    for i in range(N):
        for _ in range(ns):
            d=ds/ns
            ts=np.maximum(0.,T0+A*np.cos(PSI-np.radians(phi0_deg)))
            Mx=-r*np.sum(ts*np.sin(PSI)); My=r*np.sum(ts*np.cos(PSI))
            kap=np.array([Mx,My])/EI[i]
            ang=np.radians(tw_deg)*kap_acc
            c,sn=np.cos(ang),np.sin(ang)
            kv=np.array([kap[0]*c-kap[1]*sn, kap[0]*sn+kap[1]*c, 0.])
            p=p+(R@[0,0,1.])*d
            K=np.array([[0,-kv[2],kv[1]],[kv[2],0,-kv[0]],[-kv[1],kv[0],0]])
            R=R@(np.eye(3)+K*d+0.5*K@K*d*d)
            u,_,vt=np.linalg.svd(R); R=u@vt
            kap_acc+=np.linalg.norm(kv)*d; s_all[-1]+=d
        P.append(p.copy()); Rc.append(R.copy()); s_all.append(s_all[-1])
    return np.array(P), np.array(Rc)
def dl_of(P,Rc):
    out=[]
    for w in range(6):
        e=np.array([np.cos(PSI[w]),np.sin(PSI[w]),0.])*r
        q=P+Rc@e
        out.append(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
    return np.array(out)*1e3
def obs(A,phi,tw):
    P,Rc=forward(A,phi,tw)
    ts=np.maximum(0.,T0+A*np.cos(PSI-np.radians(phi)))
    return np.concatenate([ts, dl_of(P,Rc)])
x0=np.array([4.9,0.0,0.0]); h=np.array([1e-3,0.1,0.5])
J=np.zeros((12,3))
for k in range(3):
    xp=x0.copy(); xp[k]+=h[k]
    J[:,k]=(obs(*xp)-obs(*x0))/h[k]
U,S,Vt=np.linalg.svd(J, full_matrices=False)
print("状态 (A, φ, twist=度/弧度) -> 观测 [τ(6); ΔL(6)]")
print("  雅可比 12x3 奇异值:", np.round(S,2), " 秩 =", int((S>0.02*S[0]).sum()))
print("  条件数 %.1f"%np.linalg.cond(J))
n=np.linalg.norm(J[:6,:],axis=0); n2=np.linalg.norm(J[6:,:],axis=0)
print("\n各状态对观测的灵敏度:")
for k,nm in enumerate(("A","φ","twist")):
    print("  %-6s τ: %8.2f N/单位   ΔL: %8.3f mm/单位"%(nm,n[k],n2[k]))
print("\nΔL 的扭转签名 (与 A 的共模正交):")
d_tw=obs(4.9,0.0,0.5)-obs(4.9,0.0,0.0)
print("  ΔL 部分 =", np.round(d_tw[6:],3), " => 反对称 ±，可分离")
