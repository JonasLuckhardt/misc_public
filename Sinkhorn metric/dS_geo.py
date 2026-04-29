import torch
import numpy as np
import geomloss
import torchdiffeq
import scipy.integrate


class dS_geo:

    '''Contains formulas for computations involving the geodesic Sinkhorn metric dS in a Lagrangian discretization:
        - Computing the metric tensor gmu
        - Tracing the Hamiltonian equation to obtain geodesics for dS

        Measures are discretized by positions $ x_i $ and (fixed) masses $ m_i $ as $ \mu = \sum_{i=1}^{n_points} m_i * \delta_{x_i} $ with $ x_i \in \R^{d_dim} $. A perturbation is represented by velocities $ v_i $ as $ < \dot{\mu} , \phi > = \sum_{i=1}^{n_points} m_i * < (\nabla \phi)(x_i) , v_i > $.

        We interpret x.shape = v.shape = (n_points,d_dim), while the metric tensor has shape (n_points,d_dim,n_points,d_dim).
        The correct dual variable is given by $ p_i = m_i * \nabla ( (id - K_\mu^2)^{-1} H_\mu [mudot] )(x_i) $. Then < v , Gmu v > = < v , p >.
    '''

    def __init__(self,
                 eps=1.0,
                 n_sinkh=100,
                 sinkh_err=0.0,
                 n_inv=100,
                 n_inv_rec=1000,
                 use_geomloss=False,
                 geomloss_scaling=0.9998,
                 dtype=torch.double,
                 device=torch.device("cpu")
                ):
        self.eps=eps
        self.n_sinkh=n_sinkh
        self.sinkh_err=sinkh_err
        self.n_inv=n_inv
        self.n_inv_rec=n_inv_rec
        self.use_geomloss=use_geomloss
        self.geomloss_scaling=geomloss_scaling
        self.dtype=dtype
        self.device=device

    def __check_compatible(self,x,m,u=None):
        
        assert type(x) == torch.Tensor and x.dim() == 2
        assert u is None or (type(u) == torch.Tensor and  u.dim() == 2)
        assert m is None or (type(m) == torch.Tensor and m.dim() == 1)
        
        
        assert x.dtype == self.dtype
        assert x.device == self.device
        if not m is None:
            assert m.dtype == self.dtype
            assert m.device == self.device
            assert x.shape[0] == m.shape[0]
        if not u is None:
            assert u.dtype == self.dtype
            assert u.device == self.device
            assert x.shape == u.shape
        if m is None:
            m = torch.ones(x.shape[0],device=self.device,dtype=self.dtype) / x.shape[0]
        
        return x.shape[0], x.shape[1], m
        
    
    def __Sinkhorn_geomloss(self,x,y,mx,my):
        '''geomloss variant of Sinkhorn algorithm'''
        
        loss = geomloss.sinkhorn_samples.sinkhorn_tensorized(mx.view(1,x.shape[0]), 
                                                             x.view(1,x.shape[0],x.shape[1]), 
                                                             my.view(1,y.shape[0]), 
                                                             y.view(1,y.shape[0],y.shape[1]), 
                                                             p=2, 
                                                             blur=np.sqrt(self.eps), 
                                                             scaling=self.geomloss_scaling, 
                                                             debias=False,
                                                             potentials=True)
        return loss

    def get_kc_kmu_sq(self,x,m):
        '''compute the cost kernel, the transport kernel, and the scaling factor sq=exp(fmumu/eps)'''

        n_points, d_dim, m = self.__check_compatible(x,m)
        
        kc = torch.exp_(-(0.5*(x[:,None,:] - x[None,:,:])**2 / self.eps).sum(-1))     
        
        if self.use_geomloss: ## geomloss variant   
            sq = torch.exp(1/self.eps * self.__Sinkhorn_geomloss(x,x,m,m)[0][0,:])
            
        else:
            sq = torch.ones(n_points, dtype=self.dtype, device=self.device)
            
            for _i in range(self.n_sinkh):
                sq_old = sq
                sq = (sq / ((kc @ (sq * m)))).sqrt() ## Symmetric update step. See Feydy Thesis Alg. 3.4
                with torch.no_grad():
                    if (sq - sq_old).abs().sum() <= self.sinkh_err:
                        break
    
        kmu = sq[:,None] * kc * sq[None,:]
    
        return kc, kmu, sq

    def Sinkhorn_divergence(self,x,y,mx,my):
        if self.use_geomloss:
            return geomloss.sinkhorn_samples.sinkhorn_tensorized(mx.view(1,x.shape[0]), 
                                                             x.view(1,x.shape[0],x.shape[1]), 
                                                             my.view(1,y.shape[0]), 
                                                             y.view(1,y.shape[0],y.shape[1]), 
                                                             p=2, 
                                                             blur=np.sqrt(self.eps), 
                                                             scaling=self.geomloss_scaling, 
                                                             debias=True,
                                                             potentials=False)
        else:
            n_pointsx,d_dimx,mx = self.__check_compatible(x,mx)
            n_pointsy,d_dimy,my = self.__check_compatible(y,my)
            assert d_dimx == d_dimy
            
            sqx = torch.ones(n_pointsx, dtype=self.dtype, device=self.device)
            sqy = torch.ones(n_pointsy, dtype=self.dtype, device=self.device)
            sqxx = torch.ones(n_pointsx, dtype=self.dtype, device=self.device)
            sqyy = torch.ones(n_pointsy, dtype=self.dtype, device=self.device)
            kcxy = torch.exp_(-(0.5*(x[:,None,:] - y[None,:,:])**2 / self.eps).sum(-1)) 
            kcxx = torch.exp_(-(0.5*(x[:,None,:] - x[None,:,:])**2 / self.eps).sum(-1)) 
            kcyy = torch.exp_(-(0.5*(y[:,None,:] - y[None,:,:])**2 / self.eps).sum(-1)) 
            
            for _i in range(self.n_sinkh):
                sqx_old = sqx
                sqy_old = sqy
                sqxx_old = sqxx
                sqyy_old = sqyy
                sqxx = (sqxx / ((kcxx @ (sqxx * mx)))).sqrt() ## Symmetric update step. See Feydy Thesis Alg. 3.4
                sqyy = (sqyy / ((kcyy @ (sqyy * my)))).sqrt() ## Symmetric update step. See Feydy Thesis Alg. 3.4
                sqx = (sqx / ((kcxy @ (sqy * my)))).sqrt() ## Symmetric update step. See Feydy Thesis Alg. 3.4
                sqy = (sqy / ((kcxy.T @ (sqx * mx)))).sqrt() ## Symmetric update step. See Feydy Thesis Alg. 3.4
                with torch.no_grad():
                    if (sqx - sqx_old).abs().sum() <= self.sinkh_err and (sqy - sqy_old).abs().sum() <= self.sinkh_err and (sqxx - sqxx_old).abs().sum() <= self.sinkh_err and (sqyy - sqyy_old).abs().sum() <= self.sinkh_err:
                        break
            return (mx * (torch.log(sqx) - torch.log(sqxx))).sum() + (my * (torch.log(sqy) - torch.log(sqyy))).sum()

    def __extend_id(self,A,d_dim):
        return torch.einsum("ij,kl->ikjl",A,torch.eye(d_dim,dtype=self.dtype,device=self.device))

    def __get_Tpi(self,x,m,kmu):
        return (kmu * m[None,:]) @ x
    
    def __get_grad1_kmu(self,x,m,kmu=None):
        ## del_1 kmu (xi,yj)[v] = 1/eps * vi^T * (yj-Tpi(xi)) * kmu(xi,yj)
        ## return shape = (n,d,n)
        ### verified numerically
        
        if kmu is None:
            _,kmu,_ = self.get_kc_kmu_sq(x,m)
            
        return 1/self.eps * (x.T[None,:,:] - self.__get_Tpi(x,m,kmu)[:,:,None]) * kmu[:,None,:]
    
    def __get_grad12_kmu(self,x,m,kmu=None):
        ## del_12 kmu (xi,yj)[v,u] 
        ## = 1/eps * vi^T * uj * kmu(xi,yj)
        ##   + 1/eps**2 * vi^T * (yj - Tpi(xi)) * kmu(xi,yj) * uj^T * (xi - Tpi(yj))
        ## return shape = (n,d,n,d)
        ## does not include mass m on velocities
        
        if kmu is None:
            _,kmu,_ = self.get_kc_kmu_sq(x,m)
            
        Tpi = self.__get_Tpi(x,m,kmu)
        t1 = self.__extend_id(kmu,x.shape[1]) ## x.shape[1] = d_dim
        grads = (x.T[None,:,:,None] - Tpi[:,:,None,None])
        t2 = grads * kmu[:,None,:,None] * torch.permute(grads,dims=(2,3,0,1))
        return 1/self.eps * t1 + 1/self.eps**2 * t2
    
    
    def __eval_grad11_kmu(self,x,m,v,w,kmu):
        ## return shape = (n,n)
        ## Note that this function has no mass weights on v,w
        
        Tpi = self.__get_Tpi(x,m,kmu)
        
        vXT = v @ x.T ## !!! HAS NO m FACTOR
        wXT = w @ x.T 
    
        vTpi = (v * Tpi).sum(dim=-1)
        wTpi = (w * Tpi).sum(dim=-1)
   
    
        t1 = (vXT - vTpi[:,None]) * (wXT - wTpi[:,None]) * kmu
    
        t2 = ((vXT * wXT * kmu * m[None,:]).sum(dim=-1) - vTpi * wTpi)[:,None] * kmu   #torch.einsum("ik,ik,ik,k->i")
        
        return 1/self.eps**2 * t1 - 1/self.eps**2 * t2
    
    def __eval_grad112_kmu(self,x,m,v,w,u,kmu):
        ## v, w on first component, u on second
        ## return shape = (n,n)
        ## Note that this function has no mass weights on v,w,u
        
        Tpi = self.__get_Tpi(x,m,kmu)
    
        vXT = v @ x.T
        wXT = w @ x.T 
        XuT = x @ u.T 
    
        vTpi = (v * Tpi).sum(dim=-1)
        wTpi = (w * Tpi).sum(dim=-1)
        uTpi = (u * Tpi).sum(dim=-1)
        
        t1 = (v @ u.T) * (wXT - wTpi[:,None]) # * kmu
    
        t2 = (vXT - vTpi[:,None]) * (w @ u.T) # * kmu
    
        t3 = (vXT - vTpi[:,None]) * (wXT - wTpi[:,None]) * (XuT - uTpi[None,:]) # * kmu
    
        t4 = ((vXT * wXT * kmu * m[None,:]).sum(dim=-1) - vTpi * wTpi)[:,None] * (XuT - uTpi[None,:]) # * kmu
        
        return (1/self.eps**2 * t1 + 1/self.eps**2 * t2 + 1/self.eps**3 * t3 - 1/self.eps**3 * t4) * kmu


    def get_Gmu(self,x,m,kmu=None,n_inv=None):
        '''Compute the metric tensor Gmu.
        ## Gmu = eps/2 * m * mT * ( \grad_1 \grad_2 \kappa_mu )
        ##     = eps/2 * m * mT * ( \grad_1 \grad_2 k_mu + (\grad_1 kmu) *_mu \kappa_mu *_mu (\grad_2 k_mu))
        ## return shape = (n,d,n,d)'''

        n_points, d_dim, m = self.__check_compatible(x,m)

        if kmu is None:
            _,kmu,_ = self.get_kc_kmu_sq(x,m)
        Kmu = kmu * m[None,:]

        if n_inv is None:
            n_inv = self.n_inv
        
        t1 = self.__get_grad12_kmu(x,m,kmu)
    
        g1kmu = self.__get_grad1_kmu(x,m,kmu) ## shape = (n,d,n)
    
        kappamu = kmu
        for i in range(n_inv):
            kappamu = Kmu @ Kmu @ kappamu + kmu
        kappamu = kappamu - (n_inv - 1) / n_points * m[None,:] ## remove constant part that doesn't matter in the end
    
        ## t2 = (\grad_1 kmu) *_mu \kappa_mu *_mu (\grad_2 k_mu) using tensordot   
        t2 = torch.tensordot(g1kmu, 
                             (m[:,None,None] * torch.tensordot((kappamu * m[None,:]), 
                                                          torch.permute(g1kmu,dims=(2,0,1)),
                                                          dims=1)),
                             dims=1) 
    
        return self.eps/2 * m[:,None,None,None] * (t1 + t2) * m[None,None,:,None] 
        
    
    def eval_Gmu(self,x,m,v,kmu=None):
        '''Evaluate Gmu v. Slightly more numerically stable than get_Gmu @ v.
        return shape = (n,d)'''

        n_points, d_dim, m = self.__check_compatible(x,m,v)
        
        if kmu is None:
            _,kmu,_ = self.get_kc_kmu_sq(x,m)
        Kmu = kmu * m[None,:]
    
        t1 = torch.tensordot(self.__get_grad12_kmu(x,m,kmu),m[:,None]*v,dims=2)
        
        g1kmu = self.__get_grad1_kmu(x,m,kmu)
    
        t = torch.tensordot(torch.permute(g1kmu,dims=(2,0,1)),m[:,None]*v,dims=2)
        t = torch.linalg.lstsq(torch.eye(n_points,dtype=self.dtype,device=self.device) - Kmu @ Kmu,t).solution
        t2 = g1kmu @ (m * (Kmu @ t))
    
        return self.eps / 2 * m[:,None] * (t1 + t2)
        
    
    def eval_Gmu_inverse(self,x,m,p,kmu=None,n_inv=None):
        '''Evaluate velocity v that give p = Gmu v.
        return shape = (n,d)'''
        
        n_points, d_dim, m = self.__check_compatible(x,m,p)

        if n_inv is None:
            n_inv = self.n_inv_rec
        
        Gmu = self.get_Gmu(x,m,kmu=kmu,n_inv=n_inv)
        return torch.linalg.solve(Gmu.reshape(n_points*d_dim,n_points*d_dim),
                                  p.reshape(n_points*d_dim)
                                 ).reshape(n_points,d_dim)
    
    
    def Hamiltonian(self,x,m,p):
        return (self.eval_Gmu_inverse(x,m,p) * p).sum()
    
    def Lagrangian(self,x,m,v):
        return (self.eval_Gmu(x,m,v) * v).sum()

    
    
    
    def pdot(self,x,m,p):
        '''Compute del_x L(x,v) = - del_x H(x,p), giving the right-hand side of the Hamiltonian equation for geodesic motion.
        ## Second attempt based on Onenote -> Geodesic shooting -> "Derivative Gmu chart horizontal V2"
        return shape (n,d)'''

        n_points, d_dim, m = self.__check_compatible(x,m,p)
        eps = self.eps
        
        
        _,kmu,_ = self.get_kc_kmu_sq(x,m)
        Kmu = kmu * m[None,:]
        
        v = self.eval_Gmu_inverse(x,m,p,kmu=kmu)
    
        Tpi = self.__get_Tpi(x,m,kmu)
        g1kmu = self.__get_grad1_kmu(x,m,kmu)
        g2kmu = g1kmu.permute(2,0,1)
        g12kmu = self.__get_grad12_kmu(x,m,kmu)
    
        ## pf = (id - Kmu^2)^-1 Hmu[mudot] = (id - Kmu^2)^-1 [\grad_2 kmu *_mu v]
        ## pf = function, p = \grad pf at x
        pf = torch.linalg.lstsq(torch.eye(n_points,dtype=self.dtype,device=self.device) - Kmu @ Kmu, 
                                torch.tensordot(g1kmu.permute(2,0,1),
                                                m[:,None] * v, 
                                                dims=2)
                               ).solution
    
        
        u = v
        vXT = m[:,None] * (v @ x.T) ## shape = (n,n)
        XuT = vXT.T
    
        vTpi = m * (v * Tpi).sum(dim=-1)
        uTpi = vTpi
    
        vmg1kmu = torch.tensordot(v * m[:,None], g1kmu, dims=2) 
        g2kmumv = vmg1kmu 
    
        
        tg = (x.T[None,:,:] * vXT[:,None,:] * kmu[:,None,:] * m[None,None,:]).sum(-1) - Tpi * vTpi[:,None]
        
        ddd1 = 1/eps**2 * (m[:,None] * v @ u.T * m[None,:])[:,None,:] * (x.T[None,:,:] - Tpi[:,:,None]) # * kmu
        
        ddd2 = 1/eps**2 * (vXT [:,None,:]- vTpi[:,None,None]) * (u.T[None,:,:] * m[None,None,:]) # * kmu
    
        ddd3 = 1/eps**3 * (vXT[:,None,:] - vTpi[:,None,None]) * (x.T[None,:,:] - Tpi[:,:,None]) * (XuT[:,None,:] - uTpi[None,None,:])  # * kmu
    
        ddd4 = - 1/eps**3 * tg[:,:,None] * (XuT[:,None,:] - uTpi[None,None,:]) # * kmu
        
        t11 = 2 * ((ddd1 + ddd2 + ddd3 + ddd4) * kmu[:,None,:]).sum(-1)
    
        
        
        t121 = - 2 * m[:,None] * torch.tensordot(g12kmu, m[:,None] * v * g2kmumv[:,None], dims=2) 
    
        r1221 = 2 * torch.tensordot(g2kmu, m[:,None] * v * g2kmumv[:,None], dims=2)
    
        r1222 = - 2 * (v * torch.tensordot(g12kmu, m[:,None] * v, dims=2) ).sum(1)
    
        t1 = t11 + t121
        r1 = r1221 + r1222

    
        ###
    
        dd1 = 1/eps**2 * (x.T[None,:,:] - Tpi[:,:,None]) * ((vXT[:,:] - vTpi[:,None]) * kmu[:,:])[:,None,:]
        dd2 = - 1/eps**2 * tg[:,:,None] * kmu[:,None,:]
        t21 = torch.tensordot(dd1 + dd2, m * Kmu @ pf, dims=1) ## spotted missing m. Added that factor
    
        ## 
        
        t22 = m[:,None] * (Kmu @ pf)[:,None] * torch.tensordot(g12kmu,m[:,None] * v, dims=2)
        
        t23 = Kmu @ (Kmu @ pf)
        t23 = - m[:,None] * torch.tensordot(g12kmu, m[:,None] * v * t23[:,None], dims=2)
    
        r221 = + torch.tensordot(g2kmu, m[:,None] * v * (Kmu @ (Kmu @ pf))[:,None], dims=2) ## spotted flipped sign, changed from - to +
        
        r222 = - (v * torch.tensordot(g1kmu, m * (Kmu @ pf), dims=1) ).sum(1) 
    
        r223 = - (Kmu @ pf) * g2kmumv
    
        t2 = t21 + t22 + t23
        r2 = r221 + r222 + r223
        
        t5 = t2
        r5 = r2
                 
        
        ###
    
        t31 = m[:,None] * vmg1kmu[:,None] * torch.tensordot(g1kmu, m * pf, dims=1)
    
        t32 = m[:,None] * pf[:,None] * torch.tensordot(g1kmu, m * torch.tensordot(g2kmu, m[:,None] * v, dims=2), dims=1)
        
        r31 = - g2kmumv * (Kmu @ pf) ## r31 = r223
    
        r32 = - (Kmu @ g2kmumv) * pf
    
        t3 = t31 + t32
        r3 = r31 + r32

        
        ###
        
        t41 = 2 * m[:,None] * (Kmu @ pf)[:,None] * torch.tensordot(g1kmu, m * (Kmu @ pf), dims=1)
    
        t42 = m[:,None] * (Kmu @ (Kmu @ pf))[:,None] * torch.tensordot(g1kmu, m * pf, dims=1)
    
        t43 = m[:,None] * pf[:,None] * torch.tensordot(g1kmu, m * (Kmu @ (Kmu @ pf)), dims=1)
        
        r41 = - (Kmu @ pf) * (Kmu @ (Kmu @ pf))
    
        r42 = 2 * r41
    
        r43 = - (Kmu @ (Kmu @ (Kmu @ pf))) * pf
    
        t4 = t41 + t42 + t43
        r4 = r41 + r42 + r43

        
        ###
    
        r = r1 + r2 + r3 + r4 + r5
        t = t1 + t2 + t3 + t4 + t5
        tr = m[:,None] * torch.tensordot(g1kmu, m * torch.linalg.solve(torch.eye(n_points,device=self.device,dtype=self.dtype) + Kmu, r),dims=1)
    
        
        return eps/2 * (t + tr ) / 2





    def trace_geodesic(self,x0,m,v0,p0=None,T=1,N_tsteps=100,solver="td_dopri5", save_file=False,filename='geo',rtol=1e-5,update_method="explicit"):
        '''Solve the initial value problem of following a geodesic given an initial position and velocity. 
        Initial state can either be given as velocity v0, or initial momentum p0. If p0 is given, it overwrites v0.
        Variable 'solver' must be either 'td_XYZ', or 'sc_XYZ' for torchdiffeq-, or scipy-solvers respectively.
        
        returns xT, energy, x_path, v_path, p_path, E_path, where xT is the (autodiff)-differentiable endpoint of the traced geodesic (at time T), energy is the (autodiff)-differentiable energy of the entire path, and the _path variables give the position, velocity, momentum, and energy of the path over time.'''

        if p0 is None:
            n_points, d_dim, m = self.__check_compatible(x0,m,v0)
            p0 = self.eval_Gmu(x0,m,v0)
        else:
            n_points, d_dim, m = self.__check_compatible(x0,m,p0)
        filename=filename+"_"+str(N_tsteps)+"_"+str(int(T*100))

        
        def Hamiltonian_update(t,args):
            use_np = False
            if type(args) == np.ndarray:
                use_np=True
                args = torch.tensor(args,dtype=self.dtype,device=self.device)
                args = args.reshape(2,-1)
            x,p = args
            x = x.reshape(n_points,d_dim)
            p = p.reshape(n_points,d_dim)
            xdot = self.eval_Gmu_inverse(x,m,p) ## = v
            pdot = self.pdot(x,m,p)
            if use_np:
                return np.array([xdot.detach().numpy(),pdot.detach().numpy()]).ravel()
            else:
                return xdot, pdot

        def autodiff_update(t,args):
            use_np = False
            if type(args) == np.ndarray:
                use_np=True
                args = torch.tensor(args,dtype=self.dtype,device=self.device)
                args = args.reshape(2,-1)
            x,p = args
            x = x.reshape(n_points,d_dim)
            p = p.reshape(n_points,d_dim)
            x.requires_grad_()
            p.requires_grad_()
            Grad_x, Grad_p = torch.autograd.grad(outputs=self.Hamiltonian(x,m,p),inputs=(x,p), create_graph=True) 
            if use_np:
                return 0.5 * np.array([Grad_p.detach().numpy(),Grad_x.detach().numpy()]).ravel()
            else:
                return 0.5*Grad_p, -0.5*Grad_x 
            

        update = Hamiltonian_update
        if update_method == "autodiff":
            update = autodiff_update
        
        output=0
        if solver[:3] == "td_":
            output = torchdiffeq.odeint(update,(x0,p0),torch.linspace(0,T,N_tsteps),rtol=rtol,method=solver[3:])

        elif solver[:3] == "sc_":
            output = scipy.integrate.solve_ivp(fun=update,t_span=(0.0,1.0),
                                               y0=np.array([x0.detach().numpy(),p0.detach().numpy()]).ravel(),
                                               method=solver[3:],t_eval=np.linspace(0,1,N_tsteps),rtol=rtol)["y"]
            output = torch.tensor(output)
            output = torch.reshape(output,(2,n_points,d_dim,N_tsteps))
            output = torch.permute(output,(0,3,1,2))

        else:
            raise ValueError("invalid value for <solver>")
        
        x_path = output[0]
        p_path = output[1]
        xT = output[0][-1] ## differentiable endpoint of geodesic
        
        v_path = torch.zeros(N_tsteps,n_points,d_dim,dtype=self.dtype,device=self.device)     
        energy = 0
        
        for i in range(N_tsteps):
            v_path[i] = self.eval_Gmu_inverse(x_path[i].detach(),m,p_path[i].detach())
        E_path = (v_path * p_path).sum(dim=(1,2)).detach().numpy()
        energy = self.Hamiltonian(x0,m,p0)

        np.savez(filename,x_path.detach().numpy(),v_path.detach().numpy(),p_path.detach().numpy())
        
        return xT, energy, x_path.detach(), v_path, p_path.detach(), E_path

















