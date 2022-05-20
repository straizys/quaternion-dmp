import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import copy

class QuaternionDMP():

    def __init__(self,N_bf=20,dt=0.01):

        self.T = 1.0 
        self.dt = dt
        self.N = int(self.T/self.dt) # timesteps
        self.alphax = 1.0
        self.alphaz = 25.0
        self.betaz = 6.25
        self.N_bf = N_bf # number of basis functions
        self.tau = 1.0 # temporal scaling

        self.phase = 1.0 # initialize phase variable

    def imitate(self,demo_trajectory):
        
        t = np.linspace(0.0,self.T,demo_trajectory[:,0].shape[0])
        self.q_des = np.zeros([self.N,4])
        slerp = Slerp(t,R.from_quat(demo_trajectory[:]))
        self.q_des = slerp(np.linspace(0.0,self.T,self.N)).as_quat()

        self.dq_des_log = self.quaternion_diff(self.q_des)
        self.ddq_des_log = np.zeros(self.dq_des_log.shape)
        for d in range(3):
            self.ddq_des_log[:,d] = np.gradient(self.dq_des_log[:,d])/self.dt

        # Initial and final orientation
        self.q0 = self.q_des[0,:]
        self.dq0_log = self.dq_des_log[0,:] 
        self.ddq0_log = self.ddq_des_log[0,:]
        self.qT = self.q_des[-1,:]

        # Initialize the DMP
        self.q = copy.deepcopy(self.q0)
        self.dq_log = copy.deepcopy(self.dq0_log)
        self.ddq_log = copy.deepcopy(self.ddq0_log)

        # Centers of basis functions 
        self.c = np.ones(self.N_bf) 
        c_ = np.linspace(0,self.T,self.N_bf)
        for i in range(self.N_bf):
            self.c[i] = np.exp(-self.alphax *c_[i])

        # Widths of basis functions 
        # (as in https://github.com/studywolf/pydmps/blob/80b0a4518edf756773582cc5c40fdeee7e332169/pydmps/dmp_discrete.py#L37)
        self.h = np.ones(self.N_bf) * self.N_bf**1.5 / self.c / self.alphax

        # Evaluate the forcing term
        forcing_target = np.zeros([self.N,3]) 
        for n in range(self.N):
            forcing_target[n,:] = self.tau*self.ddq_des_log[n,:] - \
                                    self.alphaz*(self.betaz*self.logarithmic_map(
                                    self.quaternion_error(self.qT,self.q_des[n,:])) - self.dq_des_log[n,:])

        self.fit_dmp(forcing_target)
        
        return self.q_des

    def quaternion_conjugate(self,q):
        return q * np.array([-1.0,-1.0,-1.0,1.0])

    def quaternion_product(self,q1,q2):

        q12 = np.zeros(4)
        q12[-1] = q1[-1]*q2[-1] - np.dot(q1[:-1],q2[:-1])
        q12[:-1] = q1[-1]*q2[:-1] + q2[-1]*q1[:-1] + np.cross(q1[:-1],q2[:-1])
        return q12

    def quaternion_error(self,q1,q2):
        return self.quaternion_product(q1,self.quaternion_conjugate(q2))

    def exponential_map(self,r):

        theta = np.linalg.norm(r) # rotation angle
        if theta == 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0])

        n = r / np.linalg.norm(r) # rotation axis (unit vector)

        q = np.zeros(4)
        q[-1] = np.cos(theta / 2.0)
        q[:-1] = np.sin(theta/ 2.0) * n

        return q

    def logarithmic_map(self,q):

        if np.linalg.norm(q[:-1]) < np.finfo(float).eps:
            return np.zeros(3)

        n = q[:-1] / np.linalg.norm(q[:-1])
        theta = 2.0 * np.arctan2(np.linalg.norm(q[:-1]),q[-1])

        return theta*n

    def quaternion_diff(self,q):

        dq_log = np.zeros([q.shape[0], 3])
        dq_log[0,:] = self.logarithmic_map(self.quaternion_error(q[1,:], q[0,:])) / self.dt
        for n in range(1, q.shape[0]-1):
            dq_log[n,:] = self.logarithmic_map(self.quaternion_error(q[n+1,:], q[n-1,:])) / (2.0*self.dt)
        dq_log[-1,:] = self.logarithmic_map(self.quaternion_error(q[-1,:], q[-2,:])) / self.dt

        return dq_log

    def RBF(self, phase):

        if type(phase) is np.ndarray:
            return np.exp(-self.h*(phase[:,np.newaxis]-self.c)**2)
        else:
            return np.exp(-self.h*(phase-self.c)**2)

    def forcing_function_approx(self,weights,phase):

        BF = self.RBF(phase)
        if type(phase) is np.ndarray:
            return np.dot(BF,weights)*phase/np.sum(BF,axis=1)
        else:
            return np.dot(BF,weights)*phase/np.sum(BF)

    def fit_dmp(self,forcing_target):

        phase = np.exp(-self.alphax*self.tau*np.arange(0,dmp.N)/dmp.N)
        BF = self.RBF(phase)
        X = BF*phase[:,np.newaxis]/np.sum(BF,axis=1)[:,np.newaxis]
        dof = forcing_target.shape[1]

        self.weights = np.zeros([self.N_bf,dof])
        for d in range(dof):
            self.weights[:,d] = np.dot(np.linalg.pinv(X),forcing_target[:,d])

    def reset(self):
        
        self.phase = 1.0
        self.q = copy.deepcopy(self.q0)
        self.dq_log = copy.deepcopy(self.dq0_log)
        self.ddq_log = copy.deepcopy(self.ddq0_log)

    def step(self, disturbance=None):
        
        if disturbance is None:
            disturbance = np.zeros(3)
        
        self.phase += (-self.alphax * self.tau * self.phase) * (self.T/self.N)
        forcing_term = self.forcing_function_approx(self.weights,self.phase)
        
        self.ddq_log = self.alphaz*(self.betaz*self.logarithmic_map(
                self.quaternion_error(self.qT,self.q)) - self.dq_log) + forcing_term + disturbance
        self.dq_log += self.ddq_log * self.dt * self.tau
        self.q = self.quaternion_product(self.exponential_map(self.tau*self.dq_log*self.dt),self.q)
        
        return copy.deepcopy(self.q), copy.deepcopy(self.dq_log), copy.deepcopy(self.ddq_log)

    def rollout(self,tau=1.0):

        q_rollout = np.zeros([self.N,4])
        dq_log_rollout = np.zeros([self.N,3])
        ddq_log_rollout = np.zeros([self.N,3])
        q_rollout[0,:] = self.q0
        dq_log_rollout[0,:] = self.dq0_log
        ddq_log_rollout[0,:] = self.ddq0_log
        
        phase = np.exp(-self.alphax*tau*np.linspace(0.0,self.T,self.N))

        forcing_term = np.zeros([self.N,3])
        for d in range(3):
            forcing_term[:,d] = self.forcing_function_approx(self.weights[:,d],phase)

        for n in range(1,self.N):            
            ddq_log_rollout[n,:] = self.alphaz*(self.betaz*self.logarithmic_map(
                self.quaternion_error(self.qT,q_rollout[n-1,:])) - dq_log_rollout[n-1,:]) + \
                forcing_term[n,:]

            dq_log_rollout[n,:] = dq_log_rollout[n-1,:] + tau*ddq_log_rollout[n-1,:]*self.dt
            q_rollout[n,:] = self.quaternion_product(self.exponential_map(tau*dq_log_rollout[n-1,:]*self.dt),q_rollout[n-1,:])

        return q_rollout,dq_log_rollout,ddq_log_rollout


# Test

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with open('quaternion_trajectory.npy', 'rb') as f:
        demo_trajectory = np.load(f)

    # Test with valid orientation trajectory
    dmp = QuaternionDMP(N_bf = 10)
    q_des = dmp.imitate(demo_trajectory)
    q_rollout, _, _ = dmp.rollout()

    fig = plt.figure(figsize=(18,3))
    for d in range(4):
        plt.subplot(141+d)
        plt.plot(q_des[:,d],label='demo')
        plt.plot(q_rollout[:,d],'--',label='rollout')
        plt.legend()
    plt.show()
    print(np.linalg.norm(q_rollout,axis=1))

    # Test with random sequence
    dmp = QuaternionDMP(N_bf = 300)
    q_des = dmp.imitate(np.random.rand(50,4))
    q_rollout, _, _ = dmp.rollout()

    fig = plt.figure(figsize=(18,3))
    for d in range(4):
        plt.subplot(141+d)
        plt.plot(q_des[:,d],label='demo')
        plt.plot(q_rollout[:,d],'--',label='rollout')
        plt.legend()
    plt.show()
    print(np.linalg.norm(q_rollout,axis=1))

    # Test the feedback mode
    dmp = QuaternionDMP(N_bf = 10)
    q_des = dmp.imitate(demo_trajectory)
    dmp.reset()

    q_list = []
    for i in range(q_des.shape[0]):
        if i in range(10,15):
            q, _, _ = dmp.step(disturbance=30*np.random.randn(3))
        else:
            q, _, _ = dmp.step()
        q_list.append(q)

    fig = plt.figure(figsize=(18,3))
    for d in range(4):
        plt.subplot(141+d)
        plt.plot(q_des[:,d],label='demo')
        plt.plot(np.array(q_list)[:,d],'--',label='dmp')
        plt.legend()
    plt.show()
    print(np.linalg.norm(np.array(q_list),axis=1))