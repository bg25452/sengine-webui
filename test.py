import inspect

def test(a: float,c,d=1):
    print(type(a),a)

class test():
    def __init__(self):
        pass
    def t(self):
        print("t")
t = test()
m = getattr(t,"t")
m(t)

