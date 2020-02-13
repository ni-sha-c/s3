struct Runner
		solverName::String
        parameter::Real
		extraParams::Array{Float64,1}
		stateDim::Int
		timeStep::Float64
end

function primalSolver(model::Runner, u0, 
					  nSteps::Int)

		sigma, rho, beta = model.extraParams
		s = model.parameter
		dt = model.timeStep
		d = model.stateDim
		nTrj = size(u0)[2]
		
		u_trj = zeros((d,nTrj,nSteps))
		u_trj[:,:,1] = u0

        for i = 2:nSteps
		    x = u_trj[1,:,i-1]
		    y = u_trj[2,:,i-1]
		    z = u_trj[3,:,i-1]

			dx_dt = sigma.*(y - x)
            

			dy_dt = x.*(s + rho .- z) - y
			dz_dt = x.*y - beta.*z
            
            x .+= dt*dx_dt
            y .+= dt*dy_dt
            z .+= dt*dz_dt

			u_trj[1,:,i] = x
			u_trj[2,:,i] = y
			u_trj[3,:,i] = z
		end 
		return u_trj
end

lorenz63 = Runner("Lorenz 63", 0, [10., 28., 8/3], 3, 1.e-2)
u0 = rand(3,10)
u_trj = primalSolver(lorenz63, u0, 1000) 
#=
    def objective(self, fields, parameter):
        return fields[-1]

    def source(self, fields, parameter):
        sourceTerms = np.zeros_like(fields)
        sourceTerms[1] = self.dt*fields[0]
        return sourceTerms
        
    def gradientObjective(self, fields, parameter):
        dJ = np.zeros_like(fields)
        dJ[-1] = 1.0
        return dJ

    def tangentSolver(self, initFields, initPrimalFields, \
            parameter, nSteps, homogeneous=False):
        primalTrj = np.empty(shape=(nSteps, initFields.shape[0]))
        objectiveTrj = np.empty(nSteps)
        dt = self.dt
        primalTrj[0] = initPrimalFields
        objectiveTrj[0] = self.objective(primalTrj[0],parameter)
        for i in range(1, nSteps):
            primalTrj[i], objectiveTrj[i] = self.primalSolver(\
                    primalTrj[i-1], parameter, 1)
        xt, yt, zt = initFields
        sensitivity = np.dot(initFields, \
                    self.gradientObjective(primalTrj[0], parameter))/nSteps

        for i in range(nSteps):
            x, y, z = primalTrj[i]
            dxt_dt = self.sigma*(yt - xt) 
            dyt_dt = (parameter + self.rho - z)*xt - zt*x - yt 
            dzt_dt = x*yt + y*xt - self.beta*zt
            
            xt += dt*dxt_dt 
            yt += dt*dyt_dt
            zt += dt*dzt_dt
            
            finalFields = np.array([xt, yt, zt])
            if(homogeneous==False):
                finalFields += self.source(primalTrj[i],\
                        parameter)
                xt, yt, zt = finalFields

            if(i < nSteps-1):
                sensitivity += np.dot(finalFields, \
                    self.gradientObjective(primalTrj[i+1], parameter))/nSteps
        return finalFields, sensitivity
            

    def adjointSolver(self, initFields, initPrimalFields, \
            parameter, nSteps, homogeneous=False):
        rho = self.rho
        beta = self.beta
        sigma = self.sigma
        dt = self.dt
        primalTrj = np.empty(shape=(nSteps, initFields.shape[0]))
        objectiveTrj = np.empty(nSteps)

        primalTrj[0] = initPrimalFields
        objectiveTrj[0] = self.objective(primalTrj[0],parameter)
        for i in range(1, nSteps):
            primalTrj[i], objectiveTrj[i] = self.primalSolver(\
                    primalTrj[i-1], parameter, 1)
        xa, ya, za = initFields
        sensitivity = 0.
        for i in range(nSteps-1, -1, -1):
            x, y, z = primalTrj[i]
            dxa_dt = -sigma*xa + (parameter + rho - z)*ya + \
                    y*za 
            dya_dt = sigma*xa - ya + x*za 
            dza_dt = -x*ya - beta*za 
            
            xa += dt*dxa_dt 
            ya += dt*dya_dt
            za += dt*dza_dt
           
            finalFields = np.array([xa, ya, za])
            if(homogeneous==False):
                finalFields += self.gradientObjective(primalTrj[i],\
                        parameter)/nSteps
                xa, ya, za = finalFields
            if(i > 0):
                sensitivity += np.dot(finalFields, self.source(\
                    primalTrj[i-1], parameter))
        return finalFields, sensitivity
=#            



            
