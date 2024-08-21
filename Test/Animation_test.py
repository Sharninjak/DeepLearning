import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots()

t=np.linspace(0,10,100)
y=np.sin(t)
ax.axis([0,10,0,2])
ax.set_aspect(3)

while True:
    ax.plot(t,y)
    plt.pause(0.1)  #显示图形并暂停。注意用time.sleep不起作用
    ax.cla()        #清除图形
    t+=np.pi/30     #更新数据
    y=np.sin(t)
