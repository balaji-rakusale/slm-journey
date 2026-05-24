# Building a tiny autograd engine from scratch
class Value:
   def __init__(self, data):
        self.data = data
        self.grad=0.0

   def __repr__(self):
         return f"Value(data={self.data}"


   def __add__(self, other):
     return Value(self.data + other.data)


   def __mul__(self, other):
    return Value(self.data * other.data)

# Test it
a = Value(2.0)
b = Value(3.0)
c = a + b
d = a * b

print(f"a + b = {c}")
print(f"a * b = {d}")
