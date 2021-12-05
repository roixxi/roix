'''
原题目：
有2000元经费，需要采购单价为50元的若干桌子和单价为20元的若干椅子，你希望桌椅的总数尽可能的多，但要求椅子数量不少于桌子数量，且不多于桌子数量的1.5倍，那你需要怎样的一个采购方案呢？
解：要采购x1张桌子，x2把椅子

max z= x1 + x2
s.t. x1 - x2 <= 0
1.5x1 >= x2
50x1 + 20x2 <= 2000
x1, x2 >=0

在python中此类线性规划问题可用lp solver解决
scipy.optimize._linprog def linprog(c: int,
            A_ub: Optional[int] = None,
            b_ub: Optional[int] = None,
            A_eq: Optional[int] = None,
            b_eq: Optional[int] = None,
            bounds: Optional[Iterable] = None,
            method: Optional[str] = 'simplex',
            callback: Optional[Callable] = None,
            options: Optional[dict] = None) -> OptimizeResult

矩阵A：就是约束条件的系数（等号左边的系数）
矩阵B：就是约束条件的值（等号右边）
矩阵C：目标函数的系数值
'''

from scipy import  optimize as opt
import numpy as np
#参数
#c是目标函数里变量的系数
c=np.array([1,1])
#a是不等式条件的变量系数
a=np.array([[1,-1],[-1.5,1],[50,20]])
#b是是不等式条件的常数项
b=np.array([0,0,2000])
#a1，b1是等式条件的变量系数和常数项，这个例子里无等式条件,不要这两项
#a1=np.array([[1,1,1]])
#b1=np.array([7])
#限制
lim1=(0,None) #(0,None)->(0,+无穷)
lim2=(0,None)
#调用函数
ans=opt.linprog(-c,a,b,bounds=(lim1,lim2))
#输出结果
print(ans)

#注意：我们这里的应用问题，椅子不能是0.5把，所以最后应该采购37把椅子

