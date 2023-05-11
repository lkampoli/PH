import numpy as np 
import matplotlib.pyplot as plt 

p00 = np.load('Re_10595/p00.npy')
p06 = np.load('Re_10595/p06.npy')

p11 = np.load('Re_10595/p11.npy')
p12 = np.load('Re_10595/p12.npy')
p13 = np.load('Re_10595/p13.npy')

np.savetxt('p00.txt', p00, delimiter=',')
np.savetxt('p06.txt', p06, delimiter=',')
np.savetxt('p11.txt', p11, delimiter=',')
np.savetxt('p12.txt', p12, delimiter=',')
np.savetxt('p13.txt', p13, delimiter=',')

print(p11.shape)
print(p12.shape)
print(p13.shape)

plt.plot(p11[5], p11[0])
plt.plot(p12[5], p12[0])
plt.plot(p13[5], p13[0])
#plt.show()
plt.close()

plt.plot(p11[0])
plt.plot(p12[0])
plt.plot(p13[0])
#plt.show()
plt.close()
